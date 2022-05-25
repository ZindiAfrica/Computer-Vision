# Sea Turtle Rescue: Error Detection Challenge rank #1 solution combinator
# Generates the solution csv from the output of the
# zindi_sea_turtle_rescue_winner_solution_cacos_part_1.R and
# zindi_sea_turtle_rescue_winner_solution_cacos_part_2.R scripts
# The code was written by János Sávoly (a.k.a. CacoS)

pkgs <- c("tidyverse", "glue", "data.table")
ins <- lapply(pkgs, library, character.only = TRUE)

# IMPORTANT: you must set the the working.dir to the path the directory containing this file.
working.dir <- "~/ml_competiton_final_solutions/sea_turtle_error_detection_challenge/final_solution/cacos_solution_0_0442291"
setwd(working.dir)

files <- list.files(path = "models", pattern = "prediction.*.csv")

extract_score <- function(csv_file_name){
  str_extract(csv_file_name, "(0.\\d+|1)") %>% as.numeric()
  }

# At first I used the following two lines as a slightly convoluted way
# to calculate the mean missclassification error.
# score temp containes the mean accuracy (mean missclassification error = 1 - mean accuracy)
# In the winning solution I used logloss and accuracy metrics which made the
# score variable meaningless.
score_temp <- files %>% map_dbl(~extract_score(.)) %>% mean()
score <- 1 - score_temp


create_tibble <- function(csv_file_name){
  data <- read_csv(paste0("models/", csv_file_name)) %>% pull(response)
  tibble_col_name <- str_extract(csv_file_name, "(?<=\\d{1,20}_)(.*)(?=\\.csv)")
  tibble(tibble_col_name = data)
}

colnames_extractor <- function(name){
  str_extract(name, "(?<=\\d{1,20}_)(.*)(?=\\.csv)")
}

colnames <- files %>% map_chr(~colnames_extractor(.))
pred_data <- files %>% map(~create_tibble(.)) %>% reduce(cbind)
colnames(pred_data) <- colnames

pred_data %>% map(~sum(.))

test_data <-  fread("./data_input/test_data.csv", encoding = 'Latin-1')
rescue_id <- test_data[ , Rescue_ID]

solution_col_01 = character()
for (j in 1:25){
  for (i in 1:1362) {
    a = glue::glue('{rescue_id[i]} x {colnames(pred_data)[j]}')
    solution_col_01 = c(solution_col_01, a)
  }
}

solution_col_02 <- character()
for (j in 1:25){
  for (i in 1:1362) {
    a = glue::glue('{ifelse(pred_data[i, j], 1, 0)}')
    solution_col_02 = c(solution_col_02, a)
  }
}

solution <- tibble("ID" = solution_col_01,
                   "error" = solution_col_02)

solution %>% write_csv(path = glue("solution_{round(score,7)}_xgb_001.csv"))
sum(solution$error == 1) # The number of predicted errors: 253.
