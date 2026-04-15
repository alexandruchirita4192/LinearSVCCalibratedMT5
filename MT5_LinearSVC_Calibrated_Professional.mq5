#property strict
#property version   "1.00"
#property description "MT5 EA: LinearSVC calibrated ONNX classifier - Professional"
#property description "BTCUSD M15 oriented, scale-invariant 13-feature version"

#include <Trade/Trade.mqh>

#resource "ml_strategy_classifier_linearsvc_calibrated.onnx" as uchar ExtModel[]

input double InpBaseLots                 = 0.10;
input bool   InpUseConfidenceSizing      = true;
input double InpMinLotMultiplier         = 0.50;
input double InpMaxLotMultiplier         = 1.50;

input double InpEntryProbThreshold       = 0.60;
input double InpMinProbGap               = 0.20;
input bool   InpUseAtrStops              = true;
input double InpStopAtrMultiple          = 1.00;
input double InpTakeAtrMultiple          = 2.00;
input int    InpMaxBarsInTrade           = 12;
input bool   InpCloseOnOppositeSignal    = false;
input bool   InpAllowLong                = true;
input bool   InpAllowShort               = true;

input bool   InpUseHourFilter            = false;
input int    InpHourStart                = 0;
input int    InpHourEnd                  = 23;
input bool   InpUseHourSoftBias          = false;
input int    InpSoftHourStart            = 4;
input int    InpSoftHourEnd              = 18;
input double InpPreferredHourMultiplier  = 1.08;
input double InpOffHourMultiplier        = 0.94;

input bool   InpUseSpreadGuard           = false;
input bool   InpUseAdaptiveSpreadGuard   = true;
input double InpMaxSpreadPoints          = 800.0;
input double InpMaxSpreadPctOfPrice      = 0.0015;
input double InpMaxSpreadAtrFraction     = 0.20;
input double InpAdaptiveSpreadSlack      = 1.15;

input bool   InpUseCooldownAfterClose    = false;
input int    InpCooldownBars             = 2;

input bool   InpUseDailyLossGuard        = false;
input double InpDailyLossLimitMoney      = 300.0;
input bool   InpDailyLossFlatOnTrigger   = true;

input long   InpMagic                    = 26042026;
input bool   InpLog                      = false;
input bool   InpDebugLog                 = false;

const int FEATURE_COUNT = 13;
const int CLASS_COUNT   = 3;
const long EXT_INPUT_SHAPE[] = {1, FEATURE_COUNT};
const long EXT_LABEL_SHAPE[] = {1};
const long EXT_PROBA_SHAPE[] = {1, CLASS_COUNT};

enum SignalDirection
  {
   SIGNAL_SELL = -1,
   SIGNAL_FLAT =  0,
   SIGNAL_BUY  =  1
  };

struct SignalInfo
  {
   SignalDirection signal;
   double pSell;
   double pFlat;
   double pBuy;
   double bestDirectionProb;
   double probGap;
  };

CTrade trade;
long g_model_handle = INVALID_HANDLE;
datetime g_last_bar_time = 0;
int g_bars_in_trade = 0;
int g_cooldown_remaining = 0;
int g_last_history_deals_total = 0;
int g_guard_day_key = -1;
double g_guard_day_closed_pnl = 0.0;
bool g_daily_loss_guard_active = false;

int DayKey(datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
  }

bool IsNewBar()
  {
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(current_bar_time == 0)
      return false;
   if(g_last_bar_time == 0)
     {
      g_last_bar_time = current_bar_time;
      return false;
     }
   if(current_bar_time != g_last_bar_time)
     {
      g_last_bar_time = current_bar_time;
      return true;
     }
   return false;
  }

double Mean(const double &arr[], int start_shift, int count)
  {
   double sum = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
      sum += arr[i];
   return sum / count;
  }

double StdDev(const double &arr[], int start_shift, int count)
  {
   double m = Mean(arr, start_shift, count);
   double s = 0.0;
   for(int i = start_shift; i < start_shift + count; i++)
     {
      double d = arr[i] - m;
      s += d * d;
     }
   return MathSqrt(s / MathMax(count - 1, 1));
  }

double CalcATR(const MqlRates &rates[], int start_shift, int period)
  {
   double sum_tr = 0.0;
   for(int i = start_shift; i < start_shift + period; i++)
     {
      double high = rates[i].high;
      double low = rates[i].low;
      double prev_close = rates[i + 1].close;
      double tr1 = high - low;
      double tr2 = MathAbs(high - prev_close);
      double tr3 = MathAbs(low - prev_close);
      double tr = MathMax(tr1, MathMax(tr2, tr3));
      sum_tr += tr;
     }
   return sum_tr / period;
  }

bool IsHourInRange(int hour_value, int start_hour, int end_hour)
  {
   if(start_hour < 0 || start_hour > 23 || end_hour < 0 || end_hour > 23)
      return false;
   if(start_hour <= end_hour)
      return (hour_value >= start_hour && hour_value <= end_hour);
   return (hour_value >= start_hour || hour_value <= end_hour);
  }

int CurrentServerHour()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return dt.hour;
  }

double NormalizeVolumeToSymbol(double requested_lots)
  {
   double vol_min  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vol_max  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double vol_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(vol_step <= 0.0)
      vol_step = vol_min;

   double lots = MathMax(vol_min, MathMin(vol_max, requested_lots));
   lots = MathFloor(lots / vol_step) * vol_step;

   int digits = 2;
   if(vol_step > 0.0)
     {
      double tmp = vol_step;
      digits = 0;
      while(digits < 8 && MathRound(tmp) != tmp)
        {
         tmp *= 10.0;
         digits++;
        }
     }

   lots = NormalizeDouble(lots, digits);
   return MathMax(vol_min, MathMin(vol_max, lots));
  }

double GetCurrentSpreadPoints()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0)
      return 0.0;
   return (ask - bid) / point;
  }

double GetCurrentSpreadPrice()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   return ask - bid;
  }

bool SpreadAllows(double atr14_raw)
  {
   if(!InpUseSpreadGuard)
      return true;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double mid = (ask + bid) * 0.5;
   double spread_points = GetCurrentSpreadPoints();
   double spread_price  = GetCurrentSpreadPrice();

   if(mid <= 0.0)
      return false;

   if(!InpUseAdaptiveSpreadGuard)
      return (spread_points <= InpMaxSpreadPoints);

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double max_spread_price_from_points = InpMaxSpreadPoints * point;
   double max_spread_price_from_pct = mid * MathMax(0.0, InpMaxSpreadPctOfPrice);
   double max_spread_price_from_atr = atr14_raw * MathMax(0.0, InpMaxSpreadAtrFraction);

   double adaptive_limit_price = MathMin(max_spread_price_from_points, MathMin(max_spread_price_from_pct, max_spread_price_from_atr));
   adaptive_limit_price *= MathMax(1.0, InpAdaptiveSpreadSlack);

   return (spread_price <= adaptive_limit_price);
  }

bool BuildFeatureVector(matrixf &features, double &atr14_raw)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(_Symbol, _Period, 0, 80, rates) < 40)
      return false;

   double closes[], opens[];
   ArrayResize(closes, ArraySize(rates));
   ArrayResize(opens, ArraySize(rates));
   ArraySetAsSeries(closes, true);
   ArraySetAsSeries(opens, true);

   for(int i = 0; i < ArraySize(rates); i++)
     {
      closes[i] = rates[i].close;
      opens[i]  = rates[i].open;
     }

   int s = 1;
   double eps = 1e-12;
   double c = closes[s];
   double o = opens[s];
   double h = rates[s].high;
   double l = rates[s].low;

   double ret_1  = (closes[s] / (closes[s + 1] + eps)) - 1.0;
   double ret_3  = (closes[s] / (closes[s + 3] + eps)) - 1.0;
   double ret_5  = (closes[s] / (closes[s + 5] + eps)) - 1.0;
   double ret_10 = (closes[s] / (closes[s + 10] + eps)) - 1.0;

   double one_bar_returns[];
   ArrayResize(one_bar_returns, 30);
   for(int i = 0; i < 30; i++)
      one_bar_returns[i] = (closes[s + i] / (closes[s + i + 1] + eps)) - 1.0;

   double vol_10 = StdDev(one_bar_returns, 0, 10);
   double vol_20 = StdDev(one_bar_returns, 0, 20);
   double vol_ratio_10_20 = (vol_10 / (vol_20 + eps)) - 1.0;

   double sma_10 = Mean(closes, s, 10);
   double sma_20 = Mean(closes, s, 20);
   if(sma_10 == 0.0 || sma_20 == 0.0)
      return false;

   double dist_sma_10 = (c / (sma_10 + eps)) - 1.0;
   double dist_sma_20 = (c / (sma_20 + eps)) - 1.0;

   double mean_20 = Mean(closes, s, 20);
   double std_20  = StdDev(closes, s, 20);
   double zscore_20 = 0.0;
   if(std_20 > 0.0)
      zscore_20 = (c - mean_20) / std_20;

   atr14_raw = CalcATR(rates, s, 14);
   double atr_pct_14 = atr14_raw / (c + eps);
   double range_pct_1 = (h - l) / (c + eps);
   double body_pct_1 = (c - o) / (o + eps);

   features.Resize(1, FEATURE_COUNT);
   features[0][0]  = (float)ret_1;
   features[0][1]  = (float)ret_3;
   features[0][2]  = (float)ret_5;
   features[0][3]  = (float)ret_10;
   features[0][4]  = (float)vol_10;
   features[0][5]  = (float)vol_20;
   features[0][6]  = (float)vol_ratio_10_20;
   features[0][7]  = (float)dist_sma_10;
   features[0][8]  = (float)dist_sma_20;
   features[0][9]  = (float)zscore_20;
   features[0][10] = (float)atr_pct_14;
   features[0][11] = (float)range_pct_1;
   features[0][12] = (float)body_pct_1;

   return true;
  }

bool PredictProbabilities(double &pSell, double &pFlat, double &pBuy, double &atr14_raw)
  {
   matrixf x;
   if(!BuildFeatureVector(x, atr14_raw))
      return false;

   long predicted_label[1];
   matrixf probs;
   probs.Resize(1, CLASS_COUNT);

   if(!OnnxRun(g_model_handle, 0, x, predicted_label, probs))
      return false;

   pSell = probs[0][0];
   pFlat = probs[0][1];
   pBuy  = probs[0][2];
   return true;
  }

void ApplyHourSoftBias(double &pSell, double &pBuy)
  {
   if(!InpUseHourSoftBias)
      return;

   int hour_now = CurrentServerHour();
   bool preferred = IsHourInRange(hour_now, InpSoftHourStart, InpSoftHourEnd);
   double mult = preferred ? InpPreferredHourMultiplier : InpOffHourMultiplier;

   pSell *= mult;
   pBuy  *= mult;

   double maxv = MathMax(pSell, pBuy);
   if(maxv > 1.0)
     {
      pSell /= maxv;
      pBuy  /= maxv;
     }
  }

SignalInfo BuildSignalInfo(double pSell, double pFlat, double pBuy)
  {
   SignalInfo info;
   info.signal = SIGNAL_FLAT;
   info.pSell = pSell;
   info.pFlat = pFlat;
   info.pBuy  = pBuy;
   info.bestDirectionProb = MathMax(pSell, pBuy);
   info.probGap = 0.0;

   double best = pFlat;
   double second = -1.0;
   SignalDirection signal = SIGNAL_FLAT;

   if(pBuy >= pSell && pBuy > best)
     {
      second = MathMax(best, pSell);
      best = pBuy;
      signal = SIGNAL_BUY;
     }
   else if(pSell > pBuy && pSell > best)
     {
      second = MathMax(best, pBuy);
      best = pSell;
      signal = SIGNAL_SELL;
     }
   else
     {
      second = MathMax(pBuy, pSell);
      signal = SIGNAL_FLAT;
     }

   info.bestDirectionProb = best;
   info.probGap = best - second;

   if(signal == SIGNAL_BUY)
     {
      if(!InpAllowLong || pBuy < InpEntryProbThreshold || info.probGap < InpMinProbGap)
         info.signal = SIGNAL_FLAT;
      else
         info.signal = SIGNAL_BUY;
      return info;
     }

   if(signal == SIGNAL_SELL)
     {
      if(!InpAllowShort || pSell < InpEntryProbThreshold || info.probGap < InpMinProbGap)
         info.signal = SIGNAL_FLAT;
      else
         info.signal = SIGNAL_SELL;
      return info;
     }

   info.signal = SIGNAL_FLAT;
   return info;
  }

bool HasOpenPosition(long &pos_type, double &pos_price)
  {
   if(!PositionSelect(_Symbol))
      return false;
   if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic)
      return false;
   pos_type = (long)PositionGetInteger(POSITION_TYPE);
   pos_price = PositionGetDouble(POSITION_PRICE_OPEN);
   return true;
  }

void CloseOpenPosition()
  {
   if(PositionSelect(_Symbol) && (long)PositionGetInteger(POSITION_MAGIC) == InpMagic)
      trade.PositionClose(_Symbol);
  }

double ComputeLotSize(const SignalInfo &info)
  {
   double lots = InpBaseLots;
   if(!InpUseConfidenceSizing)
      return NormalizeVolumeToSymbol(lots);

   double strength_prob = MathMax(0.0, info.bestDirectionProb - InpEntryProbThreshold);
   double span_prob = MathMax(1e-8, 1.0 - InpEntryProbThreshold);
   double prob_score = MathMin(1.0, strength_prob / span_prob);

   double gap_score = 0.0;
   if(InpMinProbGap <= 0.0)
      gap_score = MathMin(1.0, info.probGap / 0.25);
   else
      gap_score = MathMin(1.0, MathMax(0.0, info.probGap - InpMinProbGap) / MathMax(1e-8, 0.30 - InpMinProbGap));

   double blended = 0.70 * prob_score + 0.30 * gap_score;
   blended = MathMax(0.0, MathMin(1.0, blended));

   double mult = InpMinLotMultiplier + (InpMaxLotMultiplier - InpMinLotMultiplier) * blended;
   lots *= mult;

   return NormalizeVolumeToSymbol(lots);
  }

void OpenTrade(const SignalInfo &info, double atr14_raw)
  {
   double lots = ComputeLotSize(info);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   double min_stop = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double sl_dist = MathMax(atr14_raw * InpStopAtrMultiple, min_stop);
   double tp_dist = MathMax(atr14_raw * InpTakeAtrMultiple, min_stop);

   double sl = 0.0;
   double tp = 0.0;

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(20);

   if(info.signal == SIGNAL_BUY)
     {
      if(InpUseAtrStops)
        {
         sl = ask - sl_dist;
         tp = ask + tp_dist;
        }
      if(trade.Buy(lots, _Symbol, ask, sl, tp, "LinearSVC Pro buy"))
         g_bars_in_trade = 0;
     }
   else if(info.signal == SIGNAL_SELL)
     {
      if(InpUseAtrStops)
        {
         sl = bid + sl_dist;
         tp = bid - tp_dist;
        }
      if(trade.Sell(lots, _Symbol, bid, sl, tp, "LinearSVC Pro sell"))
         g_bars_in_trade = 0;
     }
  }

void ManageExistingPosition(const SignalInfo &info)
  {
   long pos_type;
   double pos_price;
   if(!HasOpenPosition(pos_type, pos_price))
      return;

   g_bars_in_trade++;
   bool should_close = false;

   if(InpCloseOnOppositeSignal)
     {
      if(pos_type == POSITION_TYPE_BUY  && info.signal == SIGNAL_SELL)
         should_close = true;
      if(pos_type == POSITION_TYPE_SELL && info.signal == SIGNAL_BUY)
         should_close = true;
     }

   if(!should_close && g_bars_in_trade >= InpMaxBarsInTrade)
      should_close = true;

   if(should_close)
      CloseOpenPosition();
  }

void ResetDailyLossStateIfNeeded()
  {
   int today_key = DayKey(TimeCurrent());
   if(g_guard_day_key != today_key)
     {
      g_guard_day_key = today_key;
      g_guard_day_closed_pnl = 0.0;
      g_daily_loss_guard_active = false;
     }
  }

void RefreshClosedDealState()
  {
   ResetDailyLossStateIfNeeded();

   if(!HistorySelect(0, TimeCurrent()))
      return;

   int total = HistoryDealsTotal();
   if(total <= g_last_history_deals_total)
      return;

   for(int i = g_last_history_deals_total; i < total; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0)
         continue;

      string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      long magic    = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
      long entry    = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);

      if(symbol != _Symbol || magic != InpMagic || entry != DEAL_ENTRY_OUT)
         continue;

      double profit     = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
      double swap       = HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
      double commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
      double net = profit + swap + commission;

      datetime deal_time = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
      int deal_day_key = DayKey(deal_time);
      if(deal_day_key == g_guard_day_key)
         g_guard_day_closed_pnl += net;

      if(InpUseCooldownAfterClose)
         g_cooldown_remaining = InpCooldownBars;
     }

   g_last_history_deals_total = total;

   if(InpUseDailyLossGuard && !g_daily_loss_guard_active && g_guard_day_closed_pnl <= -MathAbs(InpDailyLossLimitMoney))
     {
      g_daily_loss_guard_active = true;
      if(InpDailyLossFlatOnTrigger)
         CloseOpenPosition();
     }
  }

void DecrementCooldown()
  {
   if(g_cooldown_remaining > 0)
      g_cooldown_remaining--;
  }

bool EntryGuardsAllow(double atr14_raw)
  {
   if(InpUseDailyLossGuard && g_daily_loss_guard_active)
      return false;
   if(InpUseCooldownAfterClose && g_cooldown_remaining > 0)
      return false;
   if(InpUseHourFilter)
     {
      int h = CurrentServerHour();
      if(!IsHourInRange(h, InpHourStart, InpHourEnd))
         return false;
     }
   if(!SpreadAllows(atr14_raw))
      return false;
   return true;
  }

int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);

   g_model_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(g_model_handle == INVALID_HANDLE)
      return INIT_FAILED;
   if(!OnnxSetInputShape(g_model_handle, 0, EXT_INPUT_SHAPE))
      return INIT_FAILED;
   if(!OnnxSetOutputShape(g_model_handle, 0, EXT_LABEL_SHAPE))
      return INIT_FAILED;
   if(!OnnxSetOutputShape(g_model_handle, 1, EXT_PROBA_SHAPE))
      return INIT_FAILED;

   ResetDailyLossStateIfNeeded();

   if(HistorySelect(0, TimeCurrent()))
      g_last_history_deals_total = HistoryDealsTotal();
   else
      g_last_history_deals_total = 0;

   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   if(g_model_handle != INVALID_HANDLE)
      OnnxRelease(g_model_handle);
   g_model_handle = INVALID_HANDLE;
  }

void OnTick()
  {
   if(!IsNewBar())
      return;

   RefreshClosedDealState();
   DecrementCooldown();

   double pSell = 0.0;
   double pFlat = 0.0;
   double pBuy  = 0.0;
   double atr14_raw = 0.0;

   if(!PredictProbabilities(pSell, pFlat, pBuy, atr14_raw))
      return;

   ApplyHourSoftBias(pSell, pBuy);

   SignalInfo info = BuildSignalInfo(pSell, pFlat, pBuy);
   ManageExistingPosition(info);

   long pos_type;
   double pos_price;
   if(HasOpenPosition(pos_type, pos_price))
      return;

   if(info.signal == SIGNAL_FLAT)
      return;

   if(!EntryGuardsAllow(atr14_raw))
      return;

   OpenTrade(info, atr14_raw);
  }
