# ---------------------------------------------------------
# fig_sp_gain_heatmap.R
# Change in AUROC against the EHR modality for every fusion
# architecture, transfer direction and outcome
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

ARCH_LAB <- c(
  "early:plain"     = "Early: concatenation",
  "early:blockstd"  = "Early: block-standardised",
  "early:pca"       = "Early: PCA-reduced",
  "inter:plain"     = "Intermediate: joint encoders",
  "inter:gated"     = "Intermediate: gated",
  "inter:moddrop"   = "Intermediate: modality dropout",
  "inter:lowrank"   = "Intermediate: low-rank tensor",
  "inter:bilinear"  = "Intermediate: bilinear pooling",
  "inter:tucker"    = "Intermediate: Tucker",
  "inter:crossattn" = "Intermediate: cross-attention",
  "late:mean"       = "Late: equal-weight average",
  "late:wsrc"       = "Late: weighted average")

DIR_LAB <- c(I2M = "INSPECT to MIMIC-IV",
             M2I = "MIMIC-IV to INSPECT",
             I2I = "Within INSPECT",
             M2M = "Within MIMIC-IV")

OUT_LAB <- c(death_30d = "30-Day Death",
             composite_30d = "Composite",
             cv_first = "CV Readmission")

## colour saturates beyond this; every tile prints its value
LIM <- 0.05

## ---- 2. data --------------------------------------------

g <- read.csv(file.path(data_dir, "grid_summary.csv"),
              stringsAsFactors = FALSE)

d <- g %>%
  filter(arch %in% names(ARCH_LAB)) %>%
  mutate(sig = lo > 0 | hi < 0,
         arch_lab = factor(unname(ARCH_LAB[arch]),
                           levels = rev(unname(ARCH_LAB))),
         direction = factor(unname(DIR_LAB[direction]),
                            levels = unname(DIR_LAB)),
         outcome = factor(unname(OUT_LAB[outcome]),
                          levels = unname(OUT_LAB)),
         txt = ifelse(round(gain, 3) == 0, "0.000",
                      sprintf("%+.3f", gain)),
         face = ifelse(sig, "bold", "plain"))

cat("\ncells:", nrow(d), " significant:", sum(d$sig), "\n")
print(table(sub(":.*", "", d$arch), d$sig))
cat("range of change:", sprintf("%+.3f", range(d$gain)),
    "\n")

## ---- 3. plot and save -----------------------------------

p <- ggplot(d, aes(x = outcome, y = arch_lab,
                   fill = gain)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = txt, fontface = face),
            size = 2.7, family = FONT, colour = "grey10") +
  facet_grid(. ~ direction) +
  scale_fill_gradient2(
    low = "#56B4E9", mid = "white", high = "#E69F00",
    midpoint = 0, limits = c(-LIM, LIM),
    oob = scales::squish,
    breaks = c(-0.05, -0.025, 0, 0.025, 0.05),
    labels = c("\u2264 -0.05", "-0.025", "0",
               "+0.025", "\u2265 +0.05"),
    name = "Change in AUROC") +
  labs(x = NULL, y = NULL,
       title = paste("Change in AUROC Against the EHR",
                     "Modality by Architecture, Direction",
                     "and Outcome")) +
  theme(panel.grid = element_blank(),
        strip.text = element_text(size = 9.5,
                                  face = "bold"),
        axis.text.x = element_text(size = 8.5, angle = 30,
                                   hjust = 1),
        axis.text.y = element_text(size = 9,
                                   colour = "grey15"),
        legend.position = "bottom",
        legend.title = element_text(size = 9,
                                    vjust = 0.8),
        legend.key.width = unit(2.2, "cm"),
        legend.key.height = unit(0.3, "cm"),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 8)),
        plot.title.position = "plot",
        plot.margin = margin(6, 8, 4, 6))

print(p)
save_png(p, "fig_sp_gain_heatmap", 10.5, 5.6)