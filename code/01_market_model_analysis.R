###############################################################################
#  東日本大震災前後の構造変化 ― 東京電力・関西電力のマーケットモデル
#  データ: data/SCmonthly.csv (2003年9月〜2020年10月, 月次)
#  収益率: 2003年10月〜2020年10月 (n = 205)
#
#  実行方法: プロジェクトのルートディレクトリを作業ディレクトリにして実行する
#  (RStudioでは 構造分析.Rproj を開いた状態が既定でルートになる)
###############################################################################

## ---- 0. データの読み込み -------------------------------------------------
xy <- read.csv("data/SCmonthly.csv", stringsAsFactors = FALSE)
head(xy); tail(xy)

topix <- xy$TOPIX   # 市場ポートフォリオ
tkyd  <- xy$TKYD    # 東京電力
knsd  <- xy$KNSD    # 関西電力

## ---- 1. 価格から収益率へ (対数差分) --------------------------------------
x  <- diff(log(topix))   # 市場収益率
y1 <- diff(log(tkyd))    # 東京電力の収益率
y2 <- diff(log(knsd))    # 関西電力の収益率
n  <- length(x)                       # 205
lab <- xy$ym[-1]                      # 収益率の対応月 ("3-Oct" 〜 "20-Oct")

## ---- 2. ダミー変数の作成 --------------------------------------------------
## 東日本大震災: 2011年3月11日 → 2011年3月の収益率(第90番目)以降を「震災後」
## d: 定数項シフト用, dx (= d*x): 傾き(ベータ)シフト用の交差項
q  <- which(lab == "11-Mar")          # 90
d  <- c(rep(0, q - 1), rep(1, n - q + 1))   # 定数項シフト用ダミー
dx <- d * x                                  # 傾き(ベータ)シフト用ダミー
c(pre = sum(d == 0), post = sum(d == 1))     # 89 / 116

## ---- 3. 時系列プロットの比較 ----------------------------------------------
## 月ラベル("3-Sep"等)を連続的な年(小数)に変換してx軸に使う
mo <- c(Jan=1,Feb=2,Mar=3,Apr=4,May=5,Jun=6,Jul=7,Aug=8,Sep=9,Oct=10,Nov=11,Dec=12)
pp <- strsplit(xy$ym, "-")
tt <- 2000 + as.numeric(sapply(pp, `[`, 1)) + (mo[sapply(pp, `[`, 2)] - 1)/12
tr <- tt[-1]; QK <- tr[q]

par(mfcol = c(3, 2), mar = c(3, 4, 2.5, 1))
vq <- function() abline(v = QK, col = "red", lty = 2)   # 震災の時点

## 左列: 株価水準
plot(tt, tkyd,  type="l", xlab="", ylab="株価(円)", main="東京電力 株価"); vq()
plot(tt, knsd,  type="l", xlab="", ylab="株価(円)", main="関西電力 株価"); vq()
plot(tt, topix, type="l", xlab="", ylab="指数",     main="TOPIX");         vq()

## 右列: 対数収益率
plot(tr, y1, type="l", xlab="", ylab="対数収益率", main="東京電力 収益率")
vq(); abline(h = 0, col = "gray")
plot(tr, y2, type="l", xlab="", ylab="対数収益率", main="関西電力 収益率")
vq(); abline(h = 0, col = "gray")
plot(tr, x,  type="l", xlab="", ylab="対数収益率", main="TOPIX 収益率")
vq(); abline(h = 0, col = "gray")

## 散布図(期間別の回帰直線つき)
par(mfrow = c(1, 2), mar = c(4.2, 4.2, 2.5, 1))
for (j in 1:2) {
  y <- if (j == 1) y1 else y2; nm <- if (j == 1) "東京電力" else "関西電力"
  plot(x, y, type="n", xlab="TOPIX 収益率", ylab=paste(nm, "収益率"), main=nm)
  points(x[d==0], y[d==0], col = "steelblue")
  points(x[d==1], y[d==1], col = "firebrick", pch = 19, cex = 0.7)
  abline(lm(y[d==0] ~ x[d==0]), col="steelblue", lwd=2)
  abline(lm(y[d==1] ~ x[d==1]), col="firebrick", lwd=2)
}

## ---- 4. 記述統計 -----------------------------------------------------------
desc <- function(v) c(n=length(v), mean=mean(v), sd=sd(v), min=min(v), max=max(v))
round(rbind(TOPIX=desc(x), TOPIX_pre=desc(x[d==0]), TOPIX_post=desc(x[d==1]),
            TKYD =desc(y1), TKYD_pre =desc(y1[d==0]), TKYD_post =desc(y1[d==1]),
            KNSD =desc(y2), KNSD_pre =desc(y2[d==0]), KNSD_post =desc(y2[d==1])), 4)

## ---- 5. モデル選択 (修正済み決定係数) --------------------------------------
## 注) 定数項なしモデルの summary()$adj.r.squared は非中心化TSSを用いるため、
##     定数項ありモデルと比較できない。中心化TSSで統一して計算する。
## adjR2: 分母を中心化TSSに統一した修正済み決定係数
adjR2 <- function(m, y) {
  k <- length(coef(m))
  1 - (sum(resid(m)^2)/(n - k)) / (sum((y - mean(y))^2)/(n - 1))
}
compare <- function(y) {
  fs <- list("a"            = y ~ 1,
             "a+bx"         = y ~ x,
             "a+cd+bx"      = y ~ d + x,
             "a+bx+g dx"    = y ~ x + dx,
             "a+cd+bx+g dx" = y ~ d + x + dx,
             "a+cd+g dx"    = y ~ d + dx,
             "bx+g dx"      = y ~ x + dx - 1,
             "cd+bx+g dx"   = y ~ d + x + dx - 1,
             "cd+g dx"      = y ~ d + dx - 1,
             "g dx"         = y ~ dx - 1)
  r <- t(sapply(fs, function(f) { m <- lm(f)
        c(k = length(coef(m)), adjR2 = adjR2(m, y), s = summary(m)$sigma, AIC = AIC(m)) }))
  round(r[order(-r[, "adjR2"]), ], 4)
}
compare(y1)   # 東京電力
compare(y2)   # 関西電力

## ---- 6. 主要モデルの推定 ---------------------------------------------------
t1_base <- lm(y1 ~ x);            summary(t1_base)   # 構造変化なし
t1_full <- lm(y1 ~ d + x + dx);   summary(t1_full)   # 完全モデル
t1_best <- lm(y1 ~ d + dx - 1);   summary(t1_best)   # 修正済みR^2 最大
k1_base <- lm(y2 ~ x);            summary(k1_base)
k1_full <- lm(y2 ~ d + x + dx);   summary(k1_full)
k1_best <- lm(y2 ~ d + dx - 1);   summary(k1_best)

## ---- 7. 構造変化の検定 (Chow検定 = c と g の同時検定) ----------------------
anova(t1_base, t1_full)   # 東京電力
anova(k1_base, k1_full)   # 関西電力

## 期間別に分けた推定
for (j in 1:2) { y <- if (j == 1) y1 else y2
  print(summary(lm(y[d==0] ~ x[d==0]))$coefficients)
  print(summary(lm(y[d==1] ~ x[d==1]))$coefficients) }

## ---- 8. 頑健性チェック -----------------------------------------------------
## (a) 不均一分散・系列相関に頑健な標準誤差 (White HC1 / Newey-West HAC)
## robse: HC1(不均一分散のみ頑健)とHAC(不均一分散+系列相関に頑健)の
##        標準誤差を自前実装で計算し、通常のOLS標準誤差と並べて比較する
robse <- function(m, lag = 4) {
  X <- model.matrix(m); u <- resid(m); N <- nrow(X); k <- ncol(X)
  br <- solve(crossprod(X)); S <- crossprod(X * u)
  hc <- br %*% (S * N/(N-k)) %*% br
  for (l in 1:lag) { w <- 1 - l/(lag+1)
    G <- t(X[(l+1):N, , drop=FALSE] * u[(l+1):N]) %*% (X[1:(N-l), , drop=FALSE] * u[1:(N-l)])
    S <- S + w * (G + t(G)) }
  nw <- br %*% (S * N/(N-k)) %*% br
  b <- coef(m)
  round(cbind(coef = b, se_OLS = summary(m)$coefficients[,2],
              se_HC1 = sqrt(diag(hc)), t_HC1 = b/sqrt(diag(hc)),
              se_HAC = sqrt(diag(nw)), t_HAC = b/sqrt(diag(nw)),
              p_HAC = 2*pnorm(-abs(b/sqrt(diag(nw))))), 4)
}
robse(t1_full); robse(k1_full)

## (b) 分散の変化 (残差分散の均一性は成立するか)
var.test(y1[d==1], y1[d==0]); var.test(y2[d==1], y2[d==0]); var.test(x[d==1], x[d==0])

## (c) 2011年3月(暴落月)を除外
summary(lm(y1[-q] ~ d[-q] + x[-q] + dx[-q]))
summary(lm(y2[-q] ~ d[-q] + x[-q] + dx[-q]))

## (d) 2011年3〜12月を除外
e <- q:(q+9)
summary(lm(y1[-e] ~ d[-e] + x[-e] + dx[-e]))
summary(lm(y2[-e] ~ d[-e] + x[-e] + dx[-e]))

## (e) 断点の位置をずらしたときの修正済み決定係数
for (j in 1:2) { y <- if (j == 1) y1 else y2
  cat(c("TKYD","KNSD")[j], ": ")
  for (s in -2:2) { qq <- q + s; dd <- c(rep(0, qq-1), rep(1, n-qq+1))
    cat(sprintf("%s=%.4f ", lab[qq], adjR2(lm(y ~ dd + x + I(dd*x)), y))) }
  cat("\n") }

## ---- 9. 補足: AIC によるステップワイズ ------------------------------------
library(MASS)
stepAIC(lm(y1 ~ 1), direction = "forward", scope = list(upper = ~ x + d + dx))
stepAIC(lm(y2 ~ 1), direction = "forward", scope = list(upper = ~ x + d + dx))
