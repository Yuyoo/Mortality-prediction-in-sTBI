
#------------------------#
#                        #
#    val.prob.ci.2       #  Adjusted version of Harrell's val.prob
#                        #
#------------------------#

# January 2016

# WHEN USING THIS FUNCTION, PLEASE CITE:
# Van Calster B, Nieboer D, Vergouwe Y, De Cock B, Pencina MJ, Steyerberg
# EW. A calibration hierarchy for risk models was defined: from utopia to
# empirical data. Journal of Clinical Epidemiology, in press (2016).

# Some years ago, Yvonne Vergouwe and Ewout Steyerberg adapted val.prob
# into val.prob.ci:
  # - Scaled Brier score by relating to max for average calibrated Null
  #   model
  # - Risk distribution according to outcome
  # - 0 and 1 to indicate outcome label; set with d1lab="..", d0lab=".."
  # - Labels: y axis: "Observed Frequency"; Triangle: "Grouped
  #   observations"
  # - Confidence intervals around triangles
  # - A cut-off can be plotted; set x coordinate

# In December 2015, Bavo De Cock, Daan Nieboer, and Ben Van Calster adapted
# this to val.prob.ci.2:
  # - Flexible calibration curves can be obtained using loess (default) or
  #   restricted cubic splines, with pointwise 95% confidence intervals
  # - Loess: CI can be obtained in closed form or using bootstrapping
  #   (CL.BT=T will do bootstrapping with 2000 bootstrap samples, however
  #   this will take a while)
  # - RCS: 3 to 5 knots can be used
  #     -> the knot locations will be estimated using default quantiles of
  #        x (by rcspline.eval, see help rcspline.plot and rcspline.eval)
  #     -> if estimation problems occur at the specified number of knots
  #        (nr.knots, default is 5), the analysis is repeated with
  #         nr.knots-1 until the problem has disappeared
  # - You can now adjust the plot through use of normal plot commands
  #   (cex.axis etc), and the size of the legend now has to be specified in
  #   cex.leg
  # - Label y-axis: "Observed proportion"
  # - Stats: added the Estimated Calibration Index (ECI), a statistical
  #   measure to quantify lack of calibration (Van Hoorde et al., 2015)
  # - Stats to be shown in the plot: by default we shown the calibration
  #   intercept (calibration-in-the-large), calibration slope and c-
  #   statistic. Alternatively, the user can select the statistics of
  #   choice (e.g. dostats=c("C (ROC)","R2") or dostats=c(2,3).
  # - Vectors p, y and logit no longer have to be sorted

  # Example:
  # # simulated data
  # x1 <- as.matrix(rnorm(500))
  # x2 <- as.matrix(rnorm(500))
  # x3 <- as.matrix(rnorm(500))
  # lp0=0.5*x1+1.2*x2+0.75*x3
  # p0true=exp(lp0)/(1+exp(lp0))
  # y <-rbinom(500,1,p0true)
  # data.0 <- data.frame(y,x1,x2,x3)
  #
  # # fit logistic model
  # fit.lrm <- lrm(y~x1+x2+x3,data=data.0)
  # pred.lrm <- predict(fit.lrm,type="fitted")
  #
  # # calibration plot for development data
  # val.prob.ci.2(pred.lrm,y)

val.prob.ci.2 <- function(p, y, logit, group, weights = rep(1, length(y)), normwt = F, pl = T,
                          smooth = c("loess","rcs",F), CL.smooth="fill",CL.BT=F,
                          nr.knots=5,logistic.cal = F, xlab = "Predicted probability", ylab =
                            "Observed proportion", xlim = c(-0.02, 1),ylim = c(-0.15,1), m, g, cuts, emax.lim = c(0, 1),
                          legendloc =  c(0.50 , 0.27), statloc = c(0,.85),dostats=T,roundstats=2,
                          riskdist = "predicted", cex=0.75,cex.leg = 0.75, connect.group =
                            F, connect.smooth = T, g.group = 4, evaluate = 100, nmin = 0, d0lab="0", d1lab="1", cex.d01=0.7,
                          dist.label=0.04, line.bins=-.05, dist.label2=.03, cutoff, las=1, length.seg=1,
                          y.intersp=1,col.ideal="grey",lwd.ideal=1,...)
{
  
  if(smooth[1]==F){smooth <- "F"}
  smooth <- match.arg(smooth)
  if(missing(p))
    p <- 1/(1 + exp( - logit))
  else logit <- log(p/(1 - p))
  if(length(p) != length(y))
    stop("lengths of p or logit and y do not agree")
  names(p) <- names(y) <- names(logit) <- NULL
  if(!missing(group)) {
    if(length(group) == 1 && is.logical(group) && group)
      group <- rep("", length(y))
    if(!is.factor(group))
      group <- if(is.logical(group) || is.character(group))
        as.factor(group) else cut2(group, g =
                                     g.group)
    names(group) <- NULL
    nma <- !(is.na(p + y + weights) | is.na(group))
    ng <- length(levels(group))
  }
  else {
    nma <- !is.na(p + y + weights)
    ng <- 0
  }
  logit <- logit[nma]
  y <- y[nma]
  p <- p[nma]
  if(ng > 0) {
    group <- group[nma]
    weights <- weights[nma]
    return(val.probg(p, y, group, evaluate, weights, normwt, nmin)
    )
  }
  require(rms)
  # Sort vector with probabilities
  y     <- y[order(p)]
  logit <- logit[order(p)]
  p     <- p[order(p)]


  if(length(p)>5000 & CL.smooth==T){warning("Number of observations > 5000, RCS is recommended.",immediate. = T)}
  if(length(p)>1000 & CL.BT==T){warning("Number of observations is > 1000, this could take a while...",immediate. = T)}


  if(length(unique(p)) == 1) {
    #22Sep94
    P <- mean(y)
    Intc <- log(P/(1 - P))
    n <- length(y)
    D <- -1/n
    L01 <- -2 * sum(y * logit - log(1 + exp(logit)), na.rm = T)
    L.cal <- -2 * sum(y * Intc - log(1 + exp(Intc)), na.rm = T)
    U.chisq <- L01 - L.cal
    U.p <- 1 - pchisq(U.chisq, 1)
    U <- (U.chisq - 1)/n
    Q <- D - U

    stats <- c(0, 0.5, 0, D, 0, 1, U, U.chisq, U.p, Q, mean((y - p[
      1])^2), Intc, 0, rep(abs(p[1] - P), 2))
    names(stats) <- c("Dxy", "C (ROC)", "R2", "D", "D:Chi-sq",
                      "D:p", "U", "U:Chi-sq", "U:p", "Q", "Brier",
                      "Intercept", "Slope", "Emax", "Eavg", "ECI")
    return(stats)
  }
  i <- !is.infinite(logit)
  nm <- sum(!i)
  if(nm > 0)
    warning(paste(nm, "observations deleted from logistic calibration due to probs. of 0 or 1"))
  i.2 <- i
  f.or <- lrm(y[i]~logit[i])
  f <- lrm.fit(logit[i], y[i])
  f2<-	lrm.fit(offset=logit[i], y=y[i])
  stats <- f$stats
  n <- stats["Obs"]
  predprob <- seq(emax.lim[1], emax.lim[2], by = 0.0005)
  lt <- f$coef[1] + f$coef[2] * log(predprob/(1 - predprob))
  calp <- 1/(1 + exp( - lt))
  emax <- max(abs(predprob - calp))
  if (pl) {
    plot(0.5, 0.5, xlim = xlim, ylim = ylim, type = "n", xlab = xlab,
         ylab = ylab, las=las,...)
    clip(0,1,0,1)
    abline(0, 1, lty = 1,col=col.ideal,lwd=lwd.ideal)
    do.call("clip", as.list(par()$usr))


    lt <- 1
    lw.d <- lwd.ideal
    leg <- "Ideal"
    marks <- -1
    if (logistic.cal) {
      lt <- c(lt, 1)
      lw.d <- c(lw.d,1)
      leg <- c(leg, "Logistic calibration")
      marks <- c(marks, -1)
    }
    if (smooth=="loess") {
      #Sm <- lowess(p,y,iter=0)
      Sm <- loess(y~p,degree=2)
      Sm <- data.frame(Sm$x,Sm$fitted); Sm.01 <- Sm

      if (connect.smooth==T & CL.smooth!="fill") {
        clip(0,1,0,1)
        lines(Sm, lty = 1,lwd=2)
        do.call("clip", as.list(par()$usr))
        lt <- c(lt, 1)
        lw.d <- c(lw.d,2)
        marks <- c(marks, -1)
      }else if(connect.smooth==F & CL.smooth!="fill"){
        clip(0,1,0,1)
        points(Sm)
        do.call("clip", as.list(par()$usr))
        lt <- c(lt, 0)
        lw.d <- c(lw.d,1)
        marks <- c(marks, 1)
      }
      if(CL.smooth==T | CL.smooth=="fill"){
        to.pred <- seq(min(p),max(p),length=200)
        if(CL.BT==T){
          BT.samples <- function(y,p,to.pred){
            data.1 <- cbind.data.frame(y,p)

            # REPEAT TO PREVENT BT SAMPLES WITH NA'S
            repeat{
              BT.sample.rows <- sample(1:nrow(data.1),replace=T)
              BT.sample <- data.1[BT.sample.rows,]
              loess(y~p,BT.sample) ->loess.BT
              predict(loess.BT,to.pred,type="fitted") ->pred.loess
              if(!any(is.na(pred.loess))){break}
            }
            return(pred.loess)
          }
          cat("Bootstrap samples are being generated.\n\n\n")

          replicate(2000,BT.samples(y,p,to.pred)) -> res.BT
          apply(res.BT,1,quantile,c(0.025,0.975)) -> CL.BT
          colnames(CL.BT) <- to.pred

          if(CL.smooth=="fill"){
            clip(0,1,0,1)
            polygon(x = c(to.pred, rev(to.pred)), y = c(CL.BT[2,],
                                                        rev(CL.BT[1,])),
                    col = rgb(177, 177, 177, 177, maxColorValue = 255), border = NA)
            if (connect.smooth==T) {
              lines(Sm, lty = 1,lwd=2)
              lt <- c(lt, 1)
              lw.d <- c(lw.d,2)
              marks <- c(marks, -1)
            }else if(connect.smooth==F){
              points(Sm)
              lt <- c(lt, 0)
              lw.d <- c(lw.d,1)
              marks <- c(marks, 1)
            }
            do.call("clip", as.list(par()$usr))
            leg <- c(leg, "Flexible calibration (Loess)")
          }else{

            clip(0,1,0,1)
            lines(to.pred,CL.BT[1,],lty=2,lwd=1);clip(0,1,0,1);lines(to.pred,CL.BT[2,],lty=2,lwd=1)
            do.call("clip", as.list(par()$usr))
            leg <- c(leg,"Flexible calibration (Loess)","CL flexible")
            lt <- c(lt,2)
            lw.d <- c(lw.d,1)
            marks <- c(marks,-1)
          }

        }else{
          Sm.0 <- loess(y~p,degree=2)
          predict(Sm.0,type="fitted",se=T) -> cl.loess
          clip(0,1,0,1)
          if(CL.smooth=="fill"){
            polygon(x = c(Sm.0$x, rev(Sm.0$x)), y = c(cl.loess$fit+cl.loess$se.fit*1.96,
                                                      rev(cl.loess$fit-cl.loess$se.fit*1.96)),
                    col = rgb(177, 177, 177, 177, maxColorValue = 255), border = NA)
            if (connect.smooth==T) {
              lines(Sm, lty = 1,lwd=2)
              lt <- c(lt, 1)
              lw.d <- c(lw.d,2)
              marks <- c(marks, -1)
            }else if(connect.smooth==F){
              points(Sm)
              lt <- c(lt, 0)
              lw.d <- c(lw.d,1)
              marks <- c(marks, 1)
            }
            do.call("clip", as.list(par()$usr))
            leg <- c(leg, "Flexible calibration (Loess)")
          }else{
            lines(Sm.0$x,cl.loess$fit+cl.loess$se.fit*1.96,lty=2,lwd=1)
            lines(Sm.0$x,cl.loess$fit-cl.loess$se.fit*1.96,lty=2,lwd=1)
            do.call("clip", as.list(par()$usr))
            leg <- c(leg,"Flexible calibration (Loess)","CL flexible")
            lt <- c(lt,2)
            lw.d <- c(lw.d,1)
            marks <- c(marks,-1)
          }

        }

      }else{
        leg <- c(leg, "Flexible calibration (Loess)")}
      cal.smooth <- approx(Sm.01, xout = p)$y
      eavg <- mean(abs(p - cal.smooth))
      ECI <- mean((p-cal.smooth)^2)*100
    }
    if(smooth=="rcs"){
      par(lwd=2,bty="n")
      if(!is.numeric(nr.knots)){stop("Nr.knots must be numeric.")}
      if(nr.knots==5){
        tryCatch(rcspline.plot(p,y,model="logistic",nk=5,show="prob", statloc = "none"
                               ,add=T,showknots=F,xrange=c(min(na.omit(p)),max(na.omit(p)))),error=function(e){
                                 warning("The number of knots led to estimation problems, nk will be set to 4.",immediate. = T)
                                 tryCatch(rcspline.plot(p,y,model="logistic",nk=4,show="prob", statloc = "none"
                                                        ,add=T,showknots=F,xrange=c(min(na.omit(p)),max(na.omit(p)))),error=function(e){
                                                          warning("Nk 4 also led to estimation problems, nk will be set to 3.",immediate.=T)
                                                          rcspline.plot(p,y,model="logistic",nk=3,show="prob", statloc = "none"
                                                                        ,add=T,showknots=F,xrange=c(min(na.omit(p)),max(na.omit(p))))
                                                        })
                               })
      }else if(nr.knots==4){
        tryCatch(rcspline.plot(p,y,model="logistic",nk=4,show="prob", statloc = "none"
                               ,add=T,showknots=F,xrange=c(min(na.omit(p)),max(na.omit(p)))),error=function(e){
                                 warning("The number of knots led to estimation problems, nk will be set to 3.",immediate.=T)
                                 rcspline.plot(p,y,model="logistic",nk=3,show="prob", statloc = "none"
                                               ,add=T,showknots=F,xrange=c(min(na.omit(p)),max(na.omit(p))))
                               })
      }else if(nr.knots==3){
        tryCatch(rcspline.plot(p,y,model="logistic",nk=3,show="prob", statloc = "none"
                               ,add=T,showknots=F,xrange=c(min(na.omit(p)),max(na.omit(p)))),
                 error=function(e){
                   stop("Nk = 3 led to estimation problems.")
                 })
      }else{stop(paste("Number of knots = ",nr.knots,sep="", ", only 5 >= nk >=3 is allowed."))}

      par(lwd=1,bty="o")
      leg <- c(leg,"Flexible calibration (RCS)","CL flexible")
      lt <- c(lt,1,2)
      lw.d <- c(lw.d,2,2)
      marks <- c(marks,-1,-1)
    }
    if(!missing(m) | !missing(g) | !missing(cuts)) {
      if(!missing(m))
        q <- cut2(p, m = m, levels.mean = T, digits = 7)
      else if(!missing(g))
        q <- cut2(p, g = g, levels.mean = T, digits = 7)
      else if(!missing(cuts))
        q <- cut2(p, cuts = cuts, levels.mean = T, digits = 7)
      means <- as.single(levels(q))
      prop <- tapply(y, q, function(x)mean(x, na.rm = T))
      points(means, prop, pch = 2, cex=1)
      #18.11.02: CI triangles
      ng	<-tapply(y, q, length)
      og	<-tapply(y, q, sum)
      ob	<-og/ng
      se.ob	<-sqrt(ob*(1-ob)/ng)
      g		<- length(as.single(levels(q)))

      for (i in 1:g) lines(c(means[i], means[i]), c(prop[i],min(1,prop[i]+1.96*se.ob[i])), type="l")
      for (i in 1:g) lines(c(means[i], means[i]), c(prop[i],max(0,prop[i]-1.96*se.ob[i])), type="l")

      if(connect.group) {
        lines(means, prop)
        lt <- c(lt, 1)
        lw.d <- c(lw.d,1)
      }
      else lt <- c(lt, 0)
      lw.d <- c(lw.d,0)
      leg <- c(leg, "Grouped observations")
      marks <- c(marks, 2)
    }
  }
  lr <- stats["Model L.R."]
  p.lr <- stats["P"]
  D <- (lr - 1)/n
  L01 <- -2 * sum(y * logit - logb(1 + exp(logit)), na.rm = TRUE)
  U.chisq <- L01 - f$deviance[2]
  p.U <- 1 - pchisq(U.chisq, 2)
  U <- (U.chisq - 2)/n
  Q <- D - U
  Dxy <- stats["Dxy"]
  C <- stats["C"]
  R2 <- stats["R2"]
  B <- sum((p - y)^2)/n
  # ES 15dec08 add Brier scaled
  Bmax  <- mean(y) * (1-mean(y))^2 + (1-mean(y)) * mean(y)^2
  Bscaled <- 1 - B/Bmax
  stats <- c(Dxy, C, R2, D, lr, p.lr, U, U.chisq, p.U, Q, B,
             f2$coef[1], f$coef[2], emax, Bscaled)
  names(stats) <- c("Dxy", "C (ROC)", "R2", "D", "D:Chi-sq",
                    "D:p", "U", "U:Chi-sq", "U:p", "Q", "Brier", "Intercept",
                    "Slope", "Emax", "Brier scaled")
  if(smooth=="loess")
    stats <- c(stats, c(Eavg = eavg),c(ECI = ECI))

  # Cut off definition
  if(!missing(cutoff)) {
    arrows(x0=cutoff,y0=.1,x1=cutoff,y1=-0.025,length=.15)
  }
  if(pl) {
    if(min(p)>plogis(-7) | max(p)<plogis(7)){

      lrm(y[i.2]~qlogis(p[i.2]))-> lrm.fit.1
      if(logistic.cal)  lines(p[i.2],plogis(lrm.fit.1$linear.predictors),lwd=1,lty=1)

    }else{logit <- seq(-7, 7, length = 200)
    prob <- 1/(1 + exp( - logit))
    pred.prob <- f$coef[1] + f$coef[2] * logit
    pred.prob <- 1/(1 + exp( - pred.prob))
    if(logistic.cal) lines(prob, pred.prob, lty = 1,lwd=1)
    }
    #	pc <- rep(" ", length(lt))
    #	pc[lt==0] <- "."
    lp <- legendloc
    if (!is.logical(lp)) {
      if (!is.list(lp))
        lp <- list(x = lp[1], y = lp[2])
      legend(lp, leg, lty = lt, pch = marks, cex = cex.leg, bty = "n",lwd=lw.d,
             col=c(col.ideal,rep("black",length(lt)-1)),y.intersp = y.intersp)
    }
    if(!is.logical(statloc)) {
      if(dostats[1]==T){
        stats.2 <- paste('Calibration\n',
                         '...intercept: ', sprintf(paste("%.",roundstats,"f",sep=""), stats["Intercept"]), '\n',
                         '...slope: ', sprintf(paste("%.",roundstats,"f",sep=""), stats["Slope"]), '\n',
                         'Discrimination\n',
                         '...c-statistic : ', sprintf(paste("%.",roundstats,"f",sep=""), stats["C (ROC)"]), sep = '')
        text(statloc[1], statloc[2],stats.2,pos=4,cex=cex)

      }else{
        dostats <- dostats
        leg <- format(names(stats)[dostats])	#constant length
        leg <- paste(leg, ":", format(stats[dostats], digits=roundstats), sep =
                       "")
        if(!is.list(statloc))
          statloc <- list(x = statloc[1], y = statloc[2])
        text(statloc, paste(format(names(stats[dostats])),
                            collapse = "\n"), adj = 0, cex = cex)
        text(statloc$x + (xlim[2]-xlim[1])/3 , statloc$y, paste(
          format(round(stats[dostats], digits=roundstats)), collapse =
            "\n"), adj = 1, cex = cex)
      }
    }
    if(is.character(riskdist)) {
      if(riskdist == "calibrated") {
        x <- f$coef[1] + f$coef[2] * log(p/(1 - p))
        x <- 1/(1 + exp( - x))
        x[p == 0] <- 0
        x[p == 1] <- 1
      }
      else x <- p
      bins <- seq(0, min(1,max(xlim)), length = 101)
      x <- x[x >= 0 & x <= 1]
      #08.04.01,yvon: distribution of predicted prob according to outcome
      f0	<-table(cut(x[y==0],bins))
      f1	<-table(cut(x[y==1],bins))
      j0	<-f0 > 0
      j1	<-f1 > 0
      bins0 <-(bins[-101])[j0]
      bins1 <-(bins[-101])[j1]
      f0	<-f0[j0]
      f1	<-f1[j1]
      maxf <-max(f0,f1)
      f0	<-(0.1*f0)/maxf
      f1	<-(0.1*f1)/maxf

      segments(bins1,line.bins,bins1,length.seg*f1+line.bins)
      segments(bins0,line.bins,bins0,length.seg*-f0+line.bins)
      lines(c(min(bins0,bins1)-0.01,max(bins0,bins1)+0.01),c(line.bins,line.bins))
      text(max(bins0,bins1)+dist.label,line.bins+dist.label2,d1lab,cex=cex.d01)
      text(max(bins0,bins1)+dist.label,line.bins-dist.label2,d0lab,cex=cex.d01)

    }
  }
  stats
}
