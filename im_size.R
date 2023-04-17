# ConvTranspose2d
x <- 1
n.layers1 <- 7
# in the order of k, s, p
k <- 4
s <- 2
p <- 1

k <- 4
s <- 1
p <- 0

for (i in 1:n.layers) {
  x <- (x - 1) * s - 2*p + k
}



# Conv2d
x <- 32
n.layers2 <- 7
# in the order of k, s, p
p.vec <- c(4, 1, 0)
p.matrix <- matrix(p.vec, nrow = n.layers2, ncol = 3)

for (i in 1:n.layers) {
  x <- ceiling((x + 2 * p - k) / s) + 1
}


x <- 32

k <- 4
s <- 2
p <- 1

x <- ceiling((x + 2 * p - k) / s) + 1