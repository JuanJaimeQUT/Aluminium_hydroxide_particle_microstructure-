# Paper 2: SEM microstructure PCA, temporal evolution and KPI analysis
# Input: one worksheet per sampling date (DD-MM-YYYY in sheet name).
# Reproduces the original 59-descriptor PCA; raw Area_ bins are retained.

packages <- c('readxl','dplyr','tidyr','stringr','lubridate','purrr','ggplot2','writexl','scales')
missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) stop('Install packages: ', paste(missing_packages, collapse = ', '))
invisible(lapply(packages, library, character.only = TRUE))

INPUT <- 'SEM_Data_Microstructure.xlsx'
OUT <- 'Paper2_PCA_Results'
N_PCS_KPI <- 6
TOP_N <- 5
set.seed(42)
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

save_plot <- function(p, filename, width = 180, height = 110) {
  for (ext in c('png', 'pdf')) {
    ggsave(file.path(OUT, paste0(filename, '.', ext)), p,
           width = width, height = height, units = 'mm',
           dpi = if (ext == 'png') 600 else 300, bg = 'white')
  }
}

family <- function(x) {
  x <- str_to_lower(x)
  case_when(
    str_detect(x, 'particle_ecd|main_ecd|satellite_mean_ecd|satellite_median_ecd|dominant_satellite_size|satellite_to_main_ratio|agglomeration_dominance_index|lt5|5_10|10_22|22_44|44_88|88_150|gt150|area_') ~ 'Size hierarchy',
    str_detect(x, 'n_crystals|main_fraction|gini_area|crystal_size_heterogeneity|main_core_fraction|main_rim_fraction|main_crystal_centrality') ~ 'Crystal population',
    str_detect(x, 'coord|cycle|graph_edges|graph_density|connected|largest_cc|community|modularity') ~ 'Connectivity / topology',
    str_detect(x, 'local_density|main_circularity|main_solidity|radial_anisotropy|rim_enrichment|rim_transition|thickness|voronoi') ~ 'Structural organization',
    str_detect(x, 'texture|laplacian|fractal|roughness|curvature|smoothness|glcm|outer_to_inner|crystal_face') ~ 'Texture and spatial heterogeneity',
    TRUE ~ 'Other'
  )
}

clean_label <- function(x) {
  x <- str_replace_all(x, '_', ' ')
  x <- str_replace_all(x, c('Particle ECD um'='Particle ECD', 'Main ECD um'='Main crystal ECD',
    'N crystals'='Crystal count', 'Thickness mean px'='Mean thickness',
    'Thickness std px'='Thickness variation', 'Voronoi area mean px2'='Mean Voronoi area',
    'Fractal dimension texture'='Texture fractal dimension', 'Gini area'='Area inequality'))
  str_squish(x)
}

family_cols <- c('Size hierarchy'='#2C7BB6','Structural organization'='#9467BD',
  'Connectivity / topology'='#D7191C','Texture and spatial heterogeneity'='#FF7F00',
  'Crystal population'='#2CA02C','Other'='#7F7F7F')
plot_theme <- theme_bw(base_size = 12) + theme(panel.grid = element_blank(),
  axis.text = element_text(colour='black'), legend.position='bottom', plot.title=element_blank())

if (!file.exists(INPUT)) stop('Input not found: ', INPUT)
sheets <- excel_sheets(INPUT)
raw <- map_dfr(sheets, function(sh) {
  date_text <- str_extract(sh, '\\d{2}-\\d{2}-\\d{4}')
  if (is.na(date_text)) stop('Sheet name must include DD-MM-YYYY: ', sh)
  read_excel(INPUT, sheet = sh) %>% mutate(Day = sh, Date = dmy(date_text))
})
if (anyNA(raw$Date)) stop('Could not parse worksheet dates')
raw <- raw %>% mutate(Day_num = as.numeric(Date - min(Date)))
meta <- raw %>% select(Day, Date, Day_num)
numeric_data <- raw %>% select(where(is.numeric)) %>% select(-Day_num) %>%
  mutate(across(everything(), ~ replace(.x, !is.finite(.x), NA_real_)))
keep <- colMeans(is.na(numeric_data)) < 0.30 &
  vapply(numeric_data, function(x) is.finite(sd(x, na.rm=TRUE)) && sd(x, na.rm=TRUE)>0, logical(1))
numeric_data <- numeric_data[, keep, drop=FALSE]
original <- names(numeric_data)
redundant <- unique(c('num_detected_crystals','Detected_crystals_df_len','Graph_nodes',
  'N_satellites','Satellite_fraction','Voronoi_area_std_px2','Thickness_n',
  grep('^Frac_|^FracArea_', original, value=TRUE)))
numeric_data <- numeric_data[, !names(numeric_data) %in% redundant, drop=FALSE]
name_key <- tibble(Original_name=names(numeric_data), Safe_name=make.names(names(numeric_data), unique=TRUE))
if (ncol(numeric_data) != 59) stop('Expected 59 descriptors under the original published PCA selection, but found ', ncol(numeric_data), '. Check input and preprocessing.')
names(numeric_data) <- name_key$Safe_name
rows <- complete.cases(numeric_data)
X <- numeric_data[rows,,drop=FALSE]
meta <- meta[rows,,drop=FALSE]
if (nrow(X)<3 || ncol(X)<3) stop('Insufficient complete data for PCA')
cat('Validated descriptor count: ', ncol(X), '\n', sep='')
variables <- name_key %>% mutate(Family=family(Original_name))
write.csv(variables, file.path(OUT, 'PCA_descriptors_retained.csv'), row.names=FALSE)
write.csv(data.frame(Variable=setdiff(names(raw), c(name_key$Original_name, 'Day', 'Date', 'Day_num'))), file.path(OUT, 'PCA_descriptors_not_retained.csv'), row.names=FALSE)
if (any(variables$Family=='Other')) warning('Unclassified descriptor families: ', paste(variables$Original_name[variables$Family=='Other'], collapse=', '))

pca <- prcomp(X, center=TRUE, scale.=TRUE)
var_fraction <- pca$sdev^2/sum(pca$sdev^2)
variance <- tibble(PC=seq_along(var_fraction), Variance_percent=100*var_fraction,
  Cumulative_percent=100*cumsum(var_fraction))
# Manuscript reference: PC1 32.2%, PC2 16.4%, PC3 6.8% (rounded).
benchmark <- tibble(PC=paste0('PC',1:3), Manuscript_percent=c(32.2,16.4,6.8),
  Recalculated_percent=variance$Variance_percent[1:3]) %>%
  mutate(Rounded_match=round(Recalculated_percent,1)==Manuscript_percent,
         Difference_pp=Recalculated_percent-Manuscript_percent)
write.csv(benchmark,file.path(OUT,'PCA_manuscript_variance_comparison.csv'),row.names=FALSE)
print(benchmark)
if (!all(benchmark$Rounded_match)) warning('PCA variance does not reproduce the manuscript at one decimal place. Review descriptor selection.')
loadings <- tibble::rownames_to_column(as.data.frame(pca$rotation), var='Safe_name') %>% left_join(variables, by='Safe_name')
scores <- as_tibble(pca$x) %>% bind_cols(meta)
label_pc <- function(pc) sprintf('%s (%.1f%%)', pc, variance$Variance_percent[as.integer(str_remove(pc,'PC'))])

p <- ggplot(head(variance, 40), aes(PC, Variance_percent)) + geom_line(colour='black', linewidth=.5) +
  geom_point(size=1.3) + labs(x='Principal component number',y='Variance explained (%)') + plot_theme
save_plot(p,'01_PCA_scree',170,110)

pc_names <- paste0('PC',1:3)
daily_pc <- scores %>% group_by(Day,Date,Day_num) %>% summarise(n_particles=n(),
  across(all_of(pc_names), list(mean=mean,sd=sd), .names='{.col}_{.fn}'), .groups='drop') %>% arrange(Day_num)
long_pc <- scores %>% pivot_longer(all_of(pc_names),names_to='PC',values_to='Score') %>%
  group_by(Day_num,PC) %>% summarise(n=n(),Mean=mean(Score),SD=sd(Score),
    Lower=Mean-1.96*SD/sqrt(n),Upper=Mean+1.96*SD/sqrt(n),.groups='drop')
for (pc in pc_names) {
  dd <- filter(long_pc,PC==pc)
  p <- ggplot(dd,aes(Day_num,Mean)) + geom_ribbon(aes(ymin=Lower,ymax=Upper),fill='#7AA6C2',alpha=.25) +
    geom_line(colour='#1F77B4') + geom_point(colour='#1F77B4') +
    labs(x='Day number',y=paste0(label_pc(pc),' score')) + plot_theme
  save_plot(p,paste0('02_',pc,'_daily_mean_95CI'),160,95)
}

pc_colours <- c(PC1='#E5A000',PC2='#9467BD',PC3='#2CA02C')
daily_scaled <- daily_pc %>% select(Day,Date,Day_num,ends_with('_mean')) %>%
  mutate(across(ends_with('_mean'),~ as.numeric(scale(.x))))
scaled_long <- daily_scaled %>% pivot_longer(ends_with('_mean'),names_to='PC',values_to='Z') %>%
  mutate(PC=str_remove(PC,'_mean'))
p <- ggplot(scaled_long,aes(Day_num,Z,colour=PC,shape=PC)) + geom_line(linewidth=.7) +
  geom_point(size=2.5) + scale_colour_manual(values=pc_colours,labels=vapply(pc_names,label_pc,character(1))) +
  scale_shape_manual(values=c(16,17,15),labels=vapply(pc_names,label_pc,character(1))) +
  labs(x='Day number',y='Standardised daily mean (z-score)',colour=NULL,shape=NULL) + plot_theme
save_plot(p,'03_PC1_PC3_standardised_temporal',185,110)

trajectory <- daily_pc %>% transmute(Day_num,PC1=PC1_mean,PC2=PC2_mean,PC3=PC3_mean)
for (pair in list(c('PC1','PC2'),c('PC1','PC3'),c('PC2','PC3'))) {
  p <- ggplot(trajectory,aes(.data[[pair[1]]],.data[[pair[2]]])) +
    geom_path(linewidth=.65,colour='grey50',arrow=grid::arrow(length=grid::unit(1.7,'mm'),type='closed')) +
    geom_point(aes(fill=Day_num),shape=21,size=3,colour='black') +
    scale_fill_gradient(low='#F3CA42',high='#A21D28',name='Day') +
    labs(x=label_pc(pair[1]),y=label_pc(pair[2])) + plot_theme
  save_plot(p,paste0('04_trajectory_',pair[1],'_',pair[2]),125,105)
}

loading_top <- map_dfr(pc_names,function(pc) loadings %>% transmute(PC=pc,Original_name,Family,
  Loading=.data[[pc]],Abs_loading=abs(.data[[pc]])) %>% slice_max(Abs_loading,n=TOP_N,with_ties=FALSE)) %>%
  mutate(Label=clean_label(Original_name),PC=factor(PC,levels=pc_names,labels=vapply(pc_names,label_pc,character(1))))
p <- ggplot(loading_top,aes(x=Abs_loading,y=reorder(Label,Abs_loading),fill=Family)) +
  geom_col(colour='black',linewidth=.2) + facet_wrap(~PC,ncol=1,scales='free_y') +
  scale_fill_manual(values=family_cols) + labs(x='Absolute loading',y=NULL,fill='Descriptor family') + plot_theme
save_plot(p,'05_top_loadings_PC1_PC3',195,190)

family_contribution <- map_dfr(seq_len(min(11,ncol(X))),function(i) {
  pc <- paste0('PC',i)
  tibble(PC=pc,Family=variables$Family,Contribution=pca$rotation[,i]^2) %>%
    group_by(PC,Family) %>% summarise(Contribution=sum(Contribution)*100,.groups='drop')
})
p <- ggplot(family_contribution,aes(PC,Contribution,fill=Family)) + geom_col(colour='white',linewidth=.2) +
  scale_fill_manual(values=family_cols) + labs(x='Principal component',y='Family contribution (%)',fill='Descriptor family') + plot_theme
save_plot(p,'06_descriptor_family_contributions_PC1_PC11',195,115)

kpi_m <- min(N_PCS_KPI,ncol(X))
weights_pc <- var_fraction[seq_len(kpi_m)]/sum(var_fraction[seq_len(kpi_m)])
importance <- as.numeric((pca$rotation[,seq_len(kpi_m),drop=FALSE]^2) %*% weights_pc)
metric <- function(x,day) {
  means <- tibble(x=x,day=day) %>% group_by(day) %>% summarise(n=n(),avg=mean(x),v=var(x),.groups='drop')
  grand <- mean(x)
  total <- sum((x-grand)^2)
  eta <- if (total>0) sum(means$n*(means$avg-grand)^2)/total else NA_real_
  within <- mean(means$v,na.rm=TRUE)
  snr <- if (is.finite(within) && within>0) var(means$avg)/within else NA_real_
  c(Eta2=eta,SNR=snr)
}
temporal <- map_dfr(seq_along(X),function(i) {
  z <- metric(X[[i]],meta$Day_num)
  tibble(Safe_name=names(X)[i],Eta2=z[['Eta2']],SNR=z[['SNR']])
})
rescale_safe <- function(x) {
  if (all(is.na(x)) || length(unique(x[is.finite(x)]))<2) return(rep(0,length(x)))
  x[!is.finite(x)] <- NA_real_
  replace(scales::rescale(x,to=c(0,1),na.rm=TRUE),is.na(x),0)
}
components <- variables %>% mutate(PCA_importance=importance) %>%
  left_join(temporal,by='Safe_name') %>% mutate(PCA_scaled=rescale_safe(PCA_importance),
  Eta2_scaled=rescale_safe(Eta2),SNR_scaled=rescale_safe(SNR),Label=clean_label(Original_name))
scenarios <- tibble(Scenario=c('Primary weighting','Equal weighting','PCA-dominant','Temporal-dominant','Robustness-dominant'),
  w_PCA=c(.50,1/3,.60,.35,.35),w_Eta2=c(.30,1/3,.25,.45,.25),w_SNR=c(.20,1/3,.15,.20,.40))
kpi_all <- map_dfr(seq_len(nrow(scenarios)),function(i) {
  s <- scenarios[i,]
  components %>% mutate(Scenario=s$Scenario, KPI_score=s$w_PCA*PCA_scaled+s$w_Eta2*Eta2_scaled+s$w_SNR*SNR_scaled) %>%
    arrange(desc(KPI_score),Original_name) %>% mutate(KPI_rank=row_number())
})
primary <- filter(kpi_all,Scenario=='Primary weighting')
rank_stability <- kpi_all %>% group_by(Original_name,Family,Label) %>%
  summarise(Mean_rank=mean(KPI_rank),Min_rank=min(KPI_rank),Max_rank=max(KPI_rank),Rank_range=Max_rank-Min_rank,.groups='drop') %>% arrange(Mean_rank)
for (n_top in c(5,10)) {
  dd <- head(primary,n_top)
  p <- ggplot(dd,aes(KPI_score,reorder(Label,KPI_score),fill=Family)) +
    geom_col(colour='black',linewidth=.2) + scale_fill_manual(values=family_cols) +
    labs(x='KPI score',y=NULL,fill='Descriptor family') + plot_theme
  save_plot(p,paste0('07_top_',n_top,'_KPI'),190,if(n_top==5) 95 else 135)
}

heat <- kpi_all %>% semi_join(head(primary,10),by='Original_name') %>%
  mutate(Label=factor(Label,levels=rev(head(primary,10)$Label)),
    Scenario=factor(Scenario,levels=scenarios$Scenario))
p <- ggplot(heat,aes(Scenario,Label,fill=KPI_rank)) + geom_tile(colour='white') +
  geom_text(aes(label=KPI_rank),size=3) + scale_fill_gradient(low='#F6D365',high='#4477AA') +
  labs(x=NULL,y=NULL,fill='Rank') + plot_theme + theme(axis.text.x=element_text(angle=25,hjust=1))
save_plot(p,'08_KPI_rank_sensitivity_heatmap',195,130)

rank_plot <- head(rank_stability,10) %>% mutate(Label=factor(Label, levels=rev(Label)))
p <- ggplot(rank_plot,aes(Mean_rank,Label)) +
  geom_segment(aes(x=Min_rank,xend=Max_rank),linewidth=1,colour='#4477AA') +
  geom_point(size=2.5) + labs(x='KPI rank across weighting scenarios',y=NULL) + plot_theme
save_plot(p,'09_KPI_rank_ranges',180,115)

top5 <- head(primary,5)
selected <- name_key$Safe_name[match(top5$Original_name,name_key$Original_name)]
top5_daily <- as_tibble(X[,selected,drop=FALSE]) %>% mutate(Day_num=meta$Day_num) %>%
  group_by(Day_num) %>% summarise(across(all_of(selected),mean),.groups='drop') %>%
  mutate(across(all_of(selected),~as.numeric(scale(.x)))) %>%
  pivot_longer(-Day_num,names_to='Safe_name',values_to='Z') %>%
  left_join(variables,by='Safe_name') %>% mutate(Label=factor(clean_label(Original_name),levels=clean_label(top5$Original_name)))
p <- ggplot(top5_daily,aes(Day_num,Z,colour=Label,shape=Label)) + geom_line(linewidth=.7) + geom_point(size=2.4) +
  labs(x='Day number',y='Standardised daily mean (z-score)',colour=NULL,shape=NULL) + plot_theme
save_plot(p,'10_top5_KPI_temporal_evolution',195,115)

# Combined temporal comparison with particle ECD, when available.
ecd <- variables %>% filter(str_detect(str_to_lower(Original_name),'^particle_ecd')) %>% slice_head(n=1)
if (nrow(ecd)==1) {
  ecd_daily <- tibble(Day_num=meta$Day_num,ECD=X[[ecd$Safe_name]]) %>%
    group_by(Day_num) %>% summarise(ECD=mean(ECD),.groups='drop') %>%
    mutate(Z=as.numeric(scale(ECD)),PC='Particle ECD') %>% select(Day_num,Z,PC)
  combined <- bind_rows(scaled_long,ecd_daily)
  p <- ggplot(combined,aes(Day_num,Z,colour=PC,shape=PC)) + geom_line(linewidth=.7) + geom_point(size=2.4) +
    scale_colour_manual(values=c(pc_colours,'Particle ECD'='#1F77B4')) +
    labs(x='Day number',y='Standardised daily mean (z-score)',colour=NULL,shape=NULL) + plot_theme
  save_plot(p,'11_PCA_and_particle_ECD_temporal',195,115)
}

write_xlsx(list(Variance_explained=variance,Variables_used=variables,
  Removed_redundant=data.frame(Variable=intersect(redundant,original)),
  Name_key=name_key,Loadings=loadings,Particle_PC_scores=scores,
  Daily_PC_summary=daily_pc,Daily_PC_CI=long_pc,
  Top_loadings=loading_top,Family_contributions=family_contribution,
  KPI_components=components,KPI_primary=primary,KPI_all_scenarios=kpi_all,
  KPI_rank_stability=rank_stability,Top5_KPI_daily=top5_daily),
  file.path(OUT,'Paper2_PCA_Analysis_Results.xlsx'))
saveRDS(pca,file.path(OUT,'PCA_model.rds'))
cat('Done: ',nrow(X),' particles, ',ncol(X),' descriptors, ',n_distinct(meta$Day_num),
    ' sampling days. Results in ',OUT,'\n',sep='')
