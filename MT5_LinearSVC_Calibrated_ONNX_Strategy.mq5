#property strict
#property version   "1.00"
#property description "MT5 EA: LinearSVC + calibrated probabilities ONNX classifier"
#property description "Scale-invariant 13-feature version for MT5 Strategy Tester"

#include <Trade/Trade.mqh>

#resource "ml_strategy_classifier_linearsvc_calibrated.onnx" as uchar ExtModel[]

input double InpLots                  = 0.10;
input double InpEntryProbThreshold    = 0.60;
input double InpMinProbGap            = 0.15;
input bool   InpUseAtrStops           = true;
input double InpStopAtrMultiple       = 1.00;
input double InpTakeAtrMultiple       = 2.75;
input int    InpMaxBarsInTrade        = 8;
input bool   InpCloseOnOppositeSignal = false;
input bool   InpAllowLong             = true;
input bool   InpAllowShort            = true;

input long   InpMagic                 = 26042026;
input bool   InpLog                   = false;
input bool   InpDebugLog              = false;

const int FEATURE_COUNT = 13;
const int CLASS_COUNT   = 3;
const long EXT_INPUT_SHAPE[]  = {1, FEATURE_COUNT};
const long EXT_LABEL_SHAPE[]  = {1};
const long EXT_PROBA_SHAPE[]  = {1, CLASS_COUNT};

enum SignalDirection
  {
   SIGNAL_SELL = -1,
   SIGNAL_FLAT =  0,
   SIGNAL_BUY  =  1
  };

CTrade trade;
long g_model_handle = INVALID_HANDLE;
datetime g_last_bar_time = 0;
int g_bars_in_trade = 0;

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

bool BuildFeatureVector(matrixf &features, double &atr14_raw)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(_Symbol, _Period, 0, 80, rates) < 40)
     {
      if(InpLog)
         Print("Not enough bars for features.");
      return false;
     }

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

bool PredictClassProbabilities(double &pSell, double &pFlat, double &pBuy, double &atr14_raw)
  {
   matrixf x;
   if(!BuildFeatureVector(x, atr14_raw))
      return false;

   long predicted_label[1];
   matrixf probs;
   probs.Resize(1, CLASS_COUNT);

   if(!OnnxRun(g_model_handle, 0, x, predicted_label, probs))
     {
      if(InpLog)
         Print("OnnxRun failed. Error=", GetLastError());
      return false;
     }

   pSell = probs[0][0];
   pFlat = probs[0][1];
   pBuy  = probs[0][2];

   if(InpDebugLog && InpLog)
      PrintFormat("RAW ONNX label=%d probs: sell=%.6f flat=%.6f buy=%.6f",
                  predicted_label[0], pSell, pFlat, pBuy);

   return true;
  }

SignalDirection SignalFromProbabilities(double pSell, double pFlat, double pBuy)
  {
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

   double gap = best - second;

   if(signal == SIGNAL_BUY)
     {
      if(!InpAllowLong)
         return SIGNAL_FLAT;
      if(pBuy < InpEntryProbThreshold || gap < InpMinProbGap)
         return SIGNAL_FLAT;
      return SIGNAL_BUY;
     }

   if(signal == SIGNAL_SELL)
     {
      if(!InpAllowShort)
         return SIGNAL_FLAT;
      if(pSell < InpEntryProbThreshold || gap < InpMinProbGap)
         return SIGNAL_FLAT;
      return SIGNAL_SELL;
     }

   return SIGNAL_FLAT;
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
     {
      bool ok = trade.PositionClose(_Symbol);
      if(!ok && InpLog)
         PrintFormat("PositionClose failed. retcode=%d lastError=%d",
                     trade.ResultRetcode(), GetLastError());
     }
  }

void OpenTrade(SignalDirection signal, double atr14_raw)
  {
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

   bool ok = false;

   if(signal == SIGNAL_BUY)
     {
      if(InpUseAtrStops)
        {
         sl = ask - sl_dist;
         tp = ask + tp_dist;
        }
      ok = trade.Buy(InpLots, _Symbol, ask, sl, tp, "LinearSVC buy");
      if(ok)
         g_bars_in_trade = 0;
     }
   else if(signal == SIGNAL_SELL)
     {
      if(InpUseAtrStops)
        {
         sl = bid + sl_dist;
         tp = bid - tp_dist;
        }
      ok = trade.Sell(InpLots, _Symbol, bid, sl, tp, "LinearSVC sell");
      if(ok)
         g_bars_in_trade = 0;
     }

   if(!ok && InpLog)
      PrintFormat("Order failed. retcode=%d lastError=%d", trade.ResultRetcode(), GetLastError());
  }

void ManageExistingPosition(SignalDirection signal)
  {
   long pos_type;
   double pos_price;
   if(!HasOpenPosition(pos_type, pos_price))
      return;

   g_bars_in_trade++;
   bool should_close = false;

   if(InpCloseOnOppositeSignal)
     {
      if(pos_type == POSITION_TYPE_BUY  && signal == SIGNAL_SELL)
         should_close = true;
      if(pos_type == POSITION_TYPE_SELL && signal == SIGNAL_BUY)
         should_close = true;
     }

   if(!should_close && g_bars_in_trade >= InpMaxBarsInTrade)
      should_close = true;

   if(should_close)
      CloseOpenPosition();
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

   double pSell = 0.0;
   double pFlat = 0.0;
   double pBuy  = 0.0;
   double atr14_raw = 0.0;

   if(!PredictClassProbabilities(pSell, pFlat, pBuy, atr14_raw))
      return;

   SignalDirection signal = SignalFromProbabilities(pSell, pFlat, pBuy);
   ManageExistingPosition(signal);

   long pos_type;
   double pos_price;
   if(HasOpenPosition(pos_type, pos_price))
      return;

   if(signal == SIGNAL_BUY || signal == SIGNAL_SELL)
      OpenTrade(signal, atr14_raw);
  }

double OnTester() {
  double profit = TesterStatistics(STAT_PROFIT);
  double pf = TesterStatistics(STAT_PROFIT_FACTOR);
  double recovery = TesterStatistics(STAT_RECOVERY_FACTOR);
  double dd_percent = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
  double trades = TesterStatistics(STAT_TRADES);

  // Penalty if there are too few transactions
  double trade_penalty = 1.0;
  if (trades < 20)
    trade_penalty = 0.25;
  else if (trades < 50)
    trade_penalty = 0.60;

  // Robust score, not only brut profit
  double score = 0.0;

  if (dd_percent >= 0.0)
    score =
        (profit * MathMax(pf, 0.01) * MathMax(recovery, 0.01) * trade_penalty) /
        (1.0 + dd_percent);

  return score;
}
