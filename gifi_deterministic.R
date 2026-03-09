library(Gifi)

n <- 500
idx <- 0:(n-1)

# Deterministic modulo generation (same as Py)
df_coded <- data.frame(
  workclass    = (idx * 7 + 3) %% 4 + 1,
  education    = (idx * 11 + 1) %% 5 + 1,
  marital      = (idx * 13 + 2) %% 4 + 1,
  occupation   = (idx * 17 + 5) %% 6 + 1,
  relationship = (idx * 19 + 0) %% 5 + 1,
  race         = (idx * 23 + 4) %% 4 + 1,
  sex          = (idx * 29 + 1) %% 2 + 1
)

# Use numeric directly for all
df_use <- df_coded

income <- as.numeric((idx * 31 + 7) %% 4 == 0)

# HOMALS
h_fit <- homals(df_use, ndim=2)
ev_h <- h_fit$evals
total_h <- sum(ev_h)
vaf_h <- ev_h / total_h * 100

# PRINCALS Nominal
pn_fit <- princals(df_use, ndim=2, levels="nominal")
ev_pn <- pn_fit$evals
vaf_pn <- ev_pn / sum(ev_pn) * 100

# PRINCALS Ordinal
po_fit <- princals(df_use, ndim=2, levels="ordinal")
ev_po <- po_fit$evals

# MAX DIFF
max_diff <- max(abs(pn_fit$loadings - po_fit$loadings))

# MORALS
m_fit <- morals(df_use, y=income)

cat("R_REF = {\n")
cat(sprintf("    \"homals_loss\":           %.9f,\n", h_fit$f))
cat(sprintf("    \"homals_eval_d1\":        %.9f,\n", ev_h[1]))
cat(sprintf("    \"homals_eval_d2\":        %.9f,\n", ev_h[2]))
cat(sprintf("    \"homals_vaf_d1\":         %.9f,\n", vaf_h[1]))
cat(sprintf("    \"homals_cumvaf\":         %.9f,\n", sum(vaf_h[1:2])))
cat(sprintf("    \"homals_ntel\":           %d,\n", h_fit$ntel))

cat(sprintf("    \"princals_nom_loss\":     %.9f,\n", pn_fit$f))
cat(sprintf("    \"princals_nom_eval_d1\":  %.9f,\n", ev_pn[1]))
cat(sprintf("    \"princals_nom_eval_d2\":  %.9f,\n", ev_pn[2]))
cat(sprintf("    \"princals_nom_vaf_d1\":   %.9f,\n", vaf_pn[1]))
cat(sprintf("    \"princals_nom_cumvaf\":   %.9f,\n", sum(vaf_pn[1:2])))
cat(sprintf("    \"princals_nom_ntel\":     %d,\n", pn_fit$ntel))

cat(sprintf("    \"princals_ord_loss\":     %.9f,\n", po_fit$f))
cat(sprintf("    \"princals_ord_eval_d1\":  %.9f,\n", ev_po[1]))
cat(sprintf("    \"nom_ord_max_diff\":      %.9e,\n", max_diff))

cat(sprintf("    \"morals_smc\":            %.9f,\n", m_fit$smc))
cat(sprintf("    \"morals_loss\":           %.9f,\n", m_fit$f))
cat(sprintf("    \"morals_beta_workclass\": %.9f,\n", m_fit$beta[1]))
cat(sprintf("    \"morals_beta_education\": %.9f,\n", m_fit$beta[2]))
cat(sprintf("    \"morals_beta_marital\":   %.9f,\n", m_fit$beta[3]))
cat(sprintf("    \"morals_beta_occupation\":%.9f,\n", m_fit$beta[4]))
cat(sprintf("    \"morals_ntel\":           %d,\n", m_fit$ntel))
cat("}\n")
