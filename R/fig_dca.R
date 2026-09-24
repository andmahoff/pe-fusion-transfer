# ---------------------------------------------------------
# fig_dca.R
# Decision curves for the three-modality block forest
# ---------------------------------------------------------

graphics.off()

library(ggplot2)
library(dplyr)
library(scales)
library(ragg)
library(systemfonts)

## ---- 1. house settings ----------------------------------

proj <- paste0(paste0(Sys.getenv("USERPROFILE"), "/Documents/"),
               "Data Science/SideProject/")

fig_dir <- paste0(proj, "R Graphs")
dir.create(fig_dir, recursive = TRUE,
           showWarnings = FALSE)

csv <- paste0(proj, "data/processed/",
              "validate_final_dca.csv")

FONT <- "Calibri"
if (nrow(subset(system_fonts(),
                family == FONT)) == 0) {
  message("Calibri not found - falling back to sans")
  FONT <- "sans"
}

theme_set(theme_minimal(base_size = 10,
                        base_family = FONT))

RUN_START <- Sys.time()

is_fresh <- function(fn) {
  if (!file.exists(fn)) return(FALSE)
  i <- file.info(fn)
  big <- i$size > 5000
  new <- i$mtime >= RUN_START
  big && new
}

save_fig <- function(plot, name, width = 6.2,
                     height = 4.4) {
  fn <- file.path(fig_dir, paste0(name, ".png"))
  if (file.exists(fn)) {
    gone <- unlink(fn)
    still <- file.exists(fn)
    if (gone != 0 || still) {
      message("  LOCKED: ", basename(fn),
              " - close any image viewer showing ",
              "it, then rerun")
      return(invisible(NA_character_))
    }
  }
  withCallingHandlers(
    tryCatch({
      ggsave(fn, plot, device = agg_png,
             width = width, height = height,
             dpi = 320, bg = "white")
    }, error = function(e) {
      message("  agg error: ", conditionMessage(e))
    }),
    warning = function(w) {
      message("  agg warning: ",
              conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  if (!is_fresh(fn)) {
    message("  agg produced nothing, using ",
            "grDevices::png")
    try({
      grDevices::png(fn, width = width * 320,
                     height = height * 320,
                     res = 320, type = "cairo",
                     bg = "white")
      print(plot)
      grDevices::dev.off()
    }, silent = TRUE)
  }
  if (!is_fresh(fn)) {
    message("  FAILED to write ", basename(fn))
    return(invisible(NA_character_))
  }
  fn
}

## ---- 2. Okabe-Ito; colour plus linetype -----------------

COLS <- c(Model        = "#0072B2",  # blue
          `Treat All`  = "#D55E00",  # vermillion
          `Treat None` = "#000000")  # black

LTYS <- c(Model        = "solid",
          `Treat All`  = "longdash",
          `Treat None` = "dotted")

LEVS <- c("Model", "Treat All", "Treat None")

## ---- 3. data --------------------------------------------

d <- read.csv(csv, stringsAsFactors = FALSE)

LABELS <- c(
  death_30d_inhosp = "In-Hospital Death",
  death_30d        = "30-Day Death",
  composite_30d    = "30-Day Composite Outcome",
  cv_first         = paste0("30-Day Cardiovascular ",
                            "Readmission"))

FILES <- c(
  death_30d_inhosp = "fig_dca_inhosp",
  death_30d        = "fig_dca_death30d",
  composite_30d    = "fig_dca_composite",
  cv_first         = "fig_dca_cvfirst")

## ---- 4. tight label placement ---------------------------

box_hits_curve <- function(cv, lx, ly, wx, wy) {
  inx <- abs(cv$x - lx) < wx
  if (!any(inx)) return(FALSE)
  gap <- min(abs(cv$y[inx] - ly))
  gap < wy * 1.15
}

fit_label <- function(curves, anchor_x, anchor_y,
                      xlim, ylim, half_w, half_h,
                      prefer_above = TRUE) {

  xr <- diff(xlim)
  yr <- diff(ylim)
  wx <- half_w * xr
  wy <- half_h * yr

  base_dy <- c(0.055, 0.075, 0.095, 0.120)
  if (!prefer_above) base_dy <- -base_dy
  dy_try <- c(base_dy, -base_dy)
  dx_try <- c(0, -0.035, 0.035, -0.070, 0.070)

  best <- NULL
  best_cost <- Inf

  for (dx in dx_try) {
    for (dy in dy_try) {
      lx <- anchor_x + dx * xr
      ly <- anchor_y + dy * yr

      # keep the whole box inside the panel
      left_ok <- (lx - wx) >= xlim[1]
      right_ok <- (lx + wx) <= xlim[2]
      low_ok <- (ly - wy) >= ylim[1]
      high_ok <- (ly + wy) <= ylim[2]
      inside <- left_ok && right_ok &&
        low_ok && high_ok
      if (!inside) next

      # reject if any curve passes through the box
      hit <- FALSE
      for (cv in curves) {
        if (box_hits_curve(cv, lx, ly, wx, wy)) {
          hit <- TRUE
          break
        }
      }
      if (hit) next

      cost <- abs(dx) + 2 * abs(dy)
      closer <- cost < best_cost
      if (closer) {
        best_cost <- cost
        best <- c(lx, ly)
      }
    }
  }

  # nothing clean: fall back to a small offset
  if (is.null(best)) {
    fb <- ifelse(prefer_above, 0.07, -0.07)
    best <- c(anchor_x, anchor_y + fb * yr)
  }
  best
}

## ---- 5. one figure per outcome --------------------------

make_dca <- function(oc) {

  z <- d[d$outcome == oc, ]
  z <- z[order(z$threshold), ]
  prev <- z$prevalence[1]

  L <- bind_rows(
    data.frame(threshold = z$threshold,
               nb = z$nb_model,
               series = "Model"),
    data.frame(threshold = z$threshold,
               nb = z$nb_treat_all,
               series = "Treat All"),
    data.frame(threshold = z$threshold,
               nb = z$nb_treat_none,
               series = "Treat None"))
  L$series <- factor(L$series, levels = LEVS)

  span <- max(z$nb_model) - min(z$nb_model, 0)
  ymin <- min(0, min(z$nb_model)) - 0.32 * span
  ymax <- max(z$nb_model) + 0.12 * span
  L$nb_clip <- pmax(pmin(L$nb, ymax), ymin)

  xlim <- c(0, 0.5)
  ylim <- c(ymin, ymax)

  ybrk <- pretty(ylim, n = 6)
  keep_b <- ybrk >= ymin & ybrk <= ymax
  ybrk <- ybrk[keep_b]
  ymnr <- pretty(ylim, n = 12)
  keep_m <- ymnr >= ymin & ymnr <= ymax
  ymnr <- setdiff(ymnr[keep_m], ybrk)

  curves <- lapply(LEVS, function(s) {
    w <- L[L$series == s, ]
    list(x = w$threshold, y = w$nb_clip)
  })

  a_m <- which.min(abs(z$threshold - 0.34))
  a_n <- which.min(abs(z$threshold - 0.44))
  cutoff <- ymin + 0.10 * (ymax - ymin)
  on <- which(z$nb_treat_all > cutoff)
  a_a <- if (length(on) > 0) on[length(on)] else 3L

  # half-sizes of each label box, as panel fractions
  hw <- c(0.052, 0.058, 0.072)
  hh <- c(0.035, 0.035, 0.035)

  ay_a <- max(z$nb_treat_all[a_a], ymin)

  p_m <- fit_label(curves, z$threshold[a_m],
                   z$nb_model[a_m], xlim, ylim,
                   hw[1], hh[1], TRUE)
  p_a <- fit_label(curves, z$threshold[a_a],
                   ay_a, xlim, ylim,
                   hw[2], hh[2], FALSE)
  p_n <- fit_label(curves, z$threshold[a_n],
                   0, xlim, ylim,
                   hw[3], hh[3], FALSE)

  lab <- data.frame(
    x = c(p_m[1], p_a[1], p_n[1]),
    y = c(p_m[2], p_a[2], p_n[2]),
    series = factor(LEVS, levels = LEVS),
    lbl = LEVS)

  ttl <- paste0("Net Benefit of the Three-Modality ",
                "Model for ", LABELS[[oc]])
  ttl <- paste(strwrap(ttl, width = 54),
               collapse = "\n")

  p <- ggplot(L, aes(x = threshold, y = nb_clip,
                     colour = series,
                     linetype = series)) +
    geom_hline(yintercept = 0, colour = "grey60",
               linewidth = 0.35) +
    geom_line(linewidth = 0.9) +
    geom_label(
      data = lab,
      aes(x = x, y = y, label = lbl,
          colour = series),
      inherit.aes = FALSE,
      family = FONT, size = 3.4,
      fontface = "bold",
      label.size = 0,
      label.padding = unit(0.12, "lines"),
      label.r = unit(0, "lines"),
      fill = "white", show.legend = FALSE) +
    scale_colour_manual(values = COLS) +
    scale_linetype_manual(values = LTYS) +
    scale_x_continuous(
      "Threshold Probability",
      limits = xlim,
      breaks = seq(0, 0.5, 0.1),
      minor_breaks = seq(0.05, 0.45, 0.1),
      labels = percent_format(accuracy = 1),
      expand = expansion(mult = c(0.01, 0.02))) +
    scale_y_continuous(
      "Net Benefit",
      limits = ylim,
      breaks = ybrk,
      minor_breaks = ymnr,
      expand = expansion(mult = c(0.01, 0.02))) +
    labs(title = ttl) +
    theme(
      legend.position = "none",
      plot.title = element_text(
        face = "bold", hjust = 0.5, size = 11.5,
        lineheight = 1.15,
        margin = margin(b = 7)),
      panel.grid.major = element_line(
        colour = "grey90", linewidth = 0.3),
      panel.grid.minor = element_line(
        colour = "grey95", linewidth = 0.2),
      axis.title.x = element_text(
        size = 10, margin = margin(t = 6)),
      axis.title.y = element_text(
        size = 10, margin = margin(r = 6)),
      axis.text = element_text(size = 9,
                               colour = "grey25"),
      plot.margin = margin(7, 9, 5, 7))

  fn <- save_fig(p, FILES[[oc]])

  if (!is.na(fn)) {
    better <- pmax(z$nb_treat_all,
                   z$nb_treat_none)
    win <- z$nb_model > better
    useful <- z$threshold[win]
    cat(sprintf(
      paste0("OK  %-22s prev %.3f  peak NB %.4f  ",
             "superior %.0f%%-%.0f%%\n"),
      basename(fn), prev, max(z$nb_model),
      100 * min(useful), 100 * max(useful)))
  } else {
    cat(sprintf("NOT WRITTEN  %s.png\n",
                FILES[[oc]]))
  }
  invisible(p)
}

for (oc in names(LABELS)) {
  if (oc %in% d$outcome) make_dca(oc)
}

graphics.off()

## ---- 6. confirm what is actually on disk ----------------

cat("\nFiles in", fig_dir, ":\n")
for (nm in FILES) {
  fn <- file.path(fig_dir, paste0(nm, ".png"))
  if (file.exists(fn)) {
    i <- file.info(fn)
    stamp <- ifelse(i$mtime >= RUN_START,
                    "UPDATED", "*** STALE ***")
    cat(sprintf("  %-24s %6.0f KB  %s  %s\n",
                paste0(nm, ".png"),
                i$size / 1024,
                format(i$mtime, "%H:%M:%S"),
                stamp))
  } else {
    cat(sprintf("  %-24s MISSING\n",
                paste0(nm, ".png")))
  }
}
cat("\nFour SEPARATE files, not a panel.\n")