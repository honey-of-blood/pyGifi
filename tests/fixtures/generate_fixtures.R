# tests/fixtures/generate_fixtures.R
# R script to generate ground-truth output from cran/Gifi

library(Gifi)
library(jsonlite)

# 1. Homals Default (Hartigan)
data(hartigan)
h_fit <- homals(hartigan, ndim = 2)
write_json(list(
  objectscores = h_fit$objectscores,
  evals = h_fit$evals,
  f = h_fit$f,
  ntel = h_fit$ntel
), "tests/fixtures/homals_hartigan.json", digits = 10)

# 2. Princals Default (ABC)
data(ABC)
# Convert ABC numeric to factors for princals ordinal
abc_ord <- as.data.frame(lapply(ABC, as.factor))
p_fit <- princals(abc_ord, ndim = 2)
write_json(list(
  objectscores = p_fit$objectscores,
  evals = p_fit$evals,
  loadings = p_fit$loadings
), "tests/fixtures/princals_abc.json", digits = 10)

# 3. Morals Default (Neumann)
data(neumann)
# neumann[,1:2] as X, neumann[,3] as y
m_fit <- morals(neumann[,1:2], neumann[,3])
write_json(list(
  smc = m_fit$smc,
  yhat = m_fit$yhat,
  ypred = m_fit$ypred
), "tests/fixtures/morals_neumann.json", digits = 10)

cat("Successfully generated R Gifi fixtures in tests/fixtures/\n")
