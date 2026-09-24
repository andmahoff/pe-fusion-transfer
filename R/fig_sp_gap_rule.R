# ---------------------------------------------------------
# fig_sp_gap_rule.R
# Late-fusion gain against the gap between the EHR and
# CTPA modalities, across the twelve direction-outcome cells
# ---------------------------------------------------------

library(ggplot2)
library(dplyr)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

home <- Sys.getenv("USERPROFILE")
fig_dir <- file.path(home, "Documents", "Data Science",
                     "SideProject", "R Graphs")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

data_dir <- file.path(home, "Documents", "Data Science",
                      "SideProject", "data", "processed")

FONT <- "Calibri"
if (nrow(subset(system_fonts(), family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10, base_family = FONT))

save_png <- function(plot, name, width, height) {
  p <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(p, plot, device = agg_png, width = width,
         height = height, units = "in", dpi = 300,
         bg = "white")
  message("saved: ", p)
}

RULE_LAB <- c("late:mean" = "Equal-weight average",
              "late:wsrc" = "Weighted average")
COL <- c("Equal-weight average" = "#0072B2",
         "Weighted average" = "#D55E00")
SHP <- c("Equal-weight average" = 16,
         "Weighted average" = 17)
LTY <- c("Equal-weight average" = "solid",
         "Weighted average" = "22")

## ---- 2. data --------------------------------------------

g <- read.csv(file.path(data_dir, "grid_summary.csv"),
              stringsAsFactors = FALSE)
gt <- read.csv(file.path(data_dir, "gap_table.csv"),
               stringsAsFactors = FALSE)

d <- g %>%
  filter(arch %in% names(RULE_LAB)) %>%
  inner_join(gt[, c("direction", "outcome", "gap")],
             by = c("direction", "outcome")) %>%
  mutate(rule = unname(RULE_LAB[arch]))

for (r in unname(RULE_LAB)) {
  x <- d[d$rule == r, ]
  ct <- cor.test(x$gap, x$gain)
  cat(sprintf("%s: n = %d, r = %.3f, p = %.5f\n", r,
              nrow(x), ct$estimate, ct$p.value))
}

fits <- do.call(rbind, lapply(split(d, d$rule), function(x) {
  m <- lm(gain ~ gap, data = x)
  xs <- range(x$gap)
  data.frame(rule = x$rule[1], gap = xs,
             gain = predict(m, data.frame(gap = xs)))
}))
ends <- fits[fits$gap == ave(fits$gap, fits$rule,
                             FUN = max), ]

## ---- 3. plot and save -----------------------------------

p <- ggplot(d, aes(x = gap, y = gain, colour = rule)) +
  geom_hline(yintercept = 0, colour = "grey45",
             linewidth = 0.4) +
  geom_line(data = fits, aes(linetype = rule),
            linewidth = 0.8) +
  geom_point(aes(shape = rule), size = 2.6) +
  geom_text(data = ends, aes(label = rule),
            hjust = 0, nudge_x = 0.004, size = 3.2,
            family = FONT, fontface = "bold",
            show.legend = FALSE) +
  scale_colour_manual(values = COL, guide = "none") +
  scale_shape_manual(values = SHP, guide = "none") +
  scale_linetype_manual(values = LTY, guide = "none") +
  scale_x_continuous(
    labels = function(x) sprintf("%.2f", x),
    expand = expansion(mult = c(0.04, 0.34))) +
  scale_y_continuous(
    breaks = seq(-0.06, 0.02, by = 0.02),
    labels = function(x) sprintf("%+.2f", x)) +
  labs(x = "Unimodal Gap (EHR AUROC Minus CTPA AUROC)",
       y = "Change in AUROC From Late Fusion",
       title = paste("Late-Fusion Gain Against the Gap",
                     "Between Modalities")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(
          colour = "grey92", linewidth = 0.3),
        axis.title = element_text(size = 9.5),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 8)),
        plot.title.position = "plot",
        plot.margin = margin(6, 10, 4, 6))

print(p)
save_png(p, "fig_sp_gap_rule", 7.2, 4.4)