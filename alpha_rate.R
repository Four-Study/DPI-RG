prob <- function(alpha, n, d = 5, eps = 0.1) {
  ## component 1
  c1 <- log(1 - 2*exp(-2*n^(1-2*alpha)*eps^2))
  if(is.na(c1)) {
    warning("c1: Increase n or decrease alpha.")
    return(0)
  }
  ## component 2
  c2 <- n*log(1 - 2*exp(-n^(2*alpha)/(2*d^2)))
  if(is.na(c2)) {
    warning("c2: Increase n or alpha.")
    return(0)
  }
  return(exp(c1 + c2))
}
## test
# prob(alpha=0.2, n=1e8)

trials <- c(2e4, 1e5, 1e6, 1e8, 1e12, 1e20)
alphas <- seq(1e-6, 0.5-1e-6, 0.002)
probs <- rep(0, length(alphas))
for (i in 1:length(trials)) {
  for (j in 1:length(probs)) {
    probs[j] <- prob(alpha = alphas[j], n = trials[i])
  }
  if (i == 1)
    plot(alphas, probs, type = "l", ylim = c(0, 1))
  else
    lines(alphas, probs, col = i)
}
legend("topright", as.character(trials), col = 1:length(trials), lty = 1)

trials <- exp(seq(10, log(1e8), 0.1))
alphas <- seq(0.1, 0.4, length.out = 7)
probs <- rep(0, length(trials))
for (i in 1:length(alphas)) {
  for (j in 1:length(probs)) {
    probs[j] <- prob(alpha = alphas[i], n = trials[j])
  }
  if (i == 1)
    plot(trials, probs, type = "l", ylim = c(0, 1))
  else
    lines(trials, probs, col = i)
}
legend("topright", as.character(alphas), col = 1:length(alphas), lty = 1)

# n <- exp(seq(5, log(.Machine$double.xmax), 0.1))
n <- exp(seq(5, 30, 0.1))
alphas <- rep(NA, length(n))
for (i in 1:length(n)) {
  sol <- optimize(f=function(alpha){prob(alpha=alpha, n=n[i])}, 
           interval = c(0, 0.5), maximum = TRUE,
           tol = .Machine$double.eps)
  alphas[i] <- sol$maximum
}
plot(log(n), alphas, type = "l")
idxs <- (alphas > 1e-6) & (alphas < 0.5 - 1e-6)
lines(log(n)[idxs], alphas[idxs], col = "red", lwd = 2)
min(alphas)
max(alphas)

probs <- apply(cbind(alphas, n), FUN = function(x){prob(x[1], x[2])}, MARGIN = 1)
probs[1:100]

x <- 1:10000
y <- 0.5 - 8/log(x)
plot(x, y, type = "l")

