# ---------------------------------------------------------
# fig_sp_architectures.R
# Change in AUROC for each fusion architecture against the
# EHR modality, INSPECT to MIMIC-IV, 30-day death
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

FAM_COL <- c(early = "#E69F00", inter = "#0072B2",
             late = "#009E73")
## filled when the interval excludes zero, hollow otherwise
SHP_SIG <- c(early = 16, inter = 17, late = 15)
SHP_NS  <- c(early = 1,  inter = 2,  late = 0)

## ---- 2. data --------------------------------------------

g <- read.csv(file.path(data_dir, "grid_summary.csv"),
              stringsAsFactors = FALSE)

d <- g %>%
  filter(direction == "I2M", outcome == "death_30d",
         arch %in% names(ARCH_LAB)) %>%
  mutate(fam = sub(":.*", "", arch),
         sig = lo > 0 | hi < 0) %>%
  arrange(gain) %>%
  mutate(lab = factor(unname(ARCH_LAB[arch]),
                      levels = unname(ARCH_LAB[arch])))
d$shp <- ifelse(d$sig, SHP_SIG[d$fam], SHP_NS[d$fam])

ref <- unique(g$ehr_ref[g$direction == "I2M" &
                        g$outcome == "death_30d"])
cat("EHR modality alone:", sprintf("%.4f", ref), "\n")
print(d[, c("arch", "auc", "gain", "lo", "hi")],
      digits = 3)

## value labels in their own column, clear of every interval
lab_x <- max(d$hi) + 0.008

## ---- 3. plot and save -----------------------------------

p <- ggplot(d, aes(y = lab, colour = fam)) +
  geom_vline(xintercept = 0, colour = "grey45",
             linewidth = 0.4) +
  geom_segment(aes(x = lo, xend = hi, yend = lab),
               linewidth = 0.6) +
  geom_point(aes(x = gain, shape = shp), size = 2.8,
             stroke = 0.9, fill = "white") +
  geom_text(aes(x = lab_x, label = sprintf("%+.4f", gain)),
            hjust = 0, size = 3, family = FONT,
            show.legend = FALSE) +
  scale_colour_manual(values = FAM_COL, guide = "none") +
  scale_shape_identity() +
  scale_x_continuous(
    breaks = seq(-0.08, 0.04, by = 0.02),
    labels = function(x) sprintf("%+.2f", x),
    expand = expansion(mult = c(0.03, 0.16))) +
  labs(x = "Change in AUROC Against the EHR Modality Alone",
       y = NULL,
       title = paste("Change in AUROC for Each Fusion",
                     "Architecture Against the EHR Modality")) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(
          colour = "grey92", linewidth = 0.3),
        axis.text.y = element_text(size = 9.5,
                                   colour = "grey15"),
        axis.title.x = element_text(
          size = 9.5, margin = margin(t = 6)),
        plot.title = element_text(
          size = 12, face = "bold", hjust = 0.5,
          margin = margin(b = 8)),
        plot.title.position = "plot",
        plot.margin = margin(6, 10, 4, 6))

print(p)
save_png(p, "fig_sp_architectures", 7.6, 4.6)