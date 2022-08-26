library(FrF2)
data <- FrF2(nruns = 64, nfactors = 4,
             factor.names = list(Optimizer = c('SGD','Adam'),
                                 Learning_rate = c('5e-3', '1e-2'),
                                 Hidden1 = c('32', '64'),
                                 Hidden2 = c('32', '64')),
             randomize=FALSE)

# Response data
y <- c(2.221, 2.009, 1.923, 2.124, 2.464, 1.879, 1.971, 1.998, # rep. 1
        2.284, 1.93, 1.98, 1.915, 2.325, 2.035, 2.016, 1.837, # rep. 1
        2.294, 1.799, 1.856, 1.907, 2.337, 1.937, 1.867, 2.105, # rep. 2
        2.161, 1.911, 1.807, 1.914, 2.293, 1.983, 2.039, 1.951, # rep. 2
        2.267, 1.832, 1.915, 1.876, 2.217, 1.97, 1.948, 1.906, # rep. 3
        2.199, 1.926, 1.969, 2.148, 2.233, 1.89, 1.895, 1.939, # rep. 3
        2.176, 1.947, 1.905, 1.839, 2.161, 1.885, 2.087, 1.953, # rep. 4
        2.161, 1.902, 1.817, 2.0, 2.317, 1.848, 1.982, 1.94) # rep. 4
dataset <- add.response(data, y)
# Make linear regression model
MAE <- lm(y ~ (Optimizer + Learning_rate + Hidden1 + Hidden2)^4, data=dataset)

# Look at the effects 
eff <- 2*MAE$coefficients

# Look at the main effects
MEPlot(MAE, main = paste('Main effects plot')) 
# Interaction plot
IAPlot(MAE, main = paste('Interaction plot matrix')) 
# Daniel plot
DanielPlot(MAE, main = paste('Normal plot'))
par(mfrow = c(2,2), mai = c(0.8, 0.35, 0.3, 0.25))
plot(MAE, pch = 19, col = 'darkgrey', sub.caption=" ")