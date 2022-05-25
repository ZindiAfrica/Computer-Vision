# Sea Turtle Rescue: Error Detection Challenge rank #1 solution part 3
# Contains the error detecting algorithms for the the following 18 columns:
# Researcher, CaptureSite, CaptureMethod, Fisher, LandingSite, Species,
# ReleaseSite, Tag_1, Tag_2, Lost_Tags, CCL_cm, CCW_cm, Weight_Kg, Sex,
# Tag_3, T_Number, PCVNumber, Expenditure, Date_Release, Status and SpecialRemarks.
# The code was written by János Sávoly (a.k.a. CacoS)

pkgs <- c("mlr", "data.table", "tidyverse", 
          "lubridate", "glue")

# IMPORTANT: you must set the the working.dir to the path the directory containing this file.
ins <- lapply(pkgs, library, character.only = TRUE)

setwd("~/ml_competiton_final_solutions/sea_turtle_error_detection_challenge/final_solution/cacos_solution_0_0442291")

cleaned_data = fread("./data_input/cleaned_data.csv", encoding = 'Latin-1')
dirty_data = fread("./data_input/dirty_data.csv", encoding = 'Latin-1')
test_data <-  fread("./data_input/test_data.csv", encoding = 'Latin-1')
rescue_id <-test_data[ , Rescue_ID]

compareNA <- function(v1,v2) {
  # This function returns TRUE wherever elements are the same, including NA's,
  # and false everywhere else.
  same <- (v1 == v2)  |  (is.na(v1) & is.na(v2))
  same[is.na(same)] <- FALSE
  return(same)
}

target <- !compareNA(cleaned_data %>% select(-Rescue_ID), dirty_data %>% select(-Rescue_ID))

data <- dirty_data %>%
  rbind(test_data) %>%
  mutate(Date_Caught = mdy(Date_Caught),
         Date_Release = mdy(Date_Release),
         comp_cap_land = compareNA(CaptureSite, LandingSite),
         comp_cap_release = compareNA(CaptureSite, ReleaseSite),
         comp_release_land = compareNA(ReleaseSite, LandingSite),
         Date_Caught_year = year(Date_Caught),
         Date_Release_year = year(Date_Release),
         Date_Caught_day = mday(Date_Caught),
         Date_Caught_wday = wday(Date_Caught),
         Date_Caught_jul = as.period(Date_Caught-ymd("1998-04-14")) %>% day(),
         TurtleCharacteristics = as.character(TurtleCharacteristics),
         Researcher = str_replace(Researcher, "researcher_", ""),
         Researcher = str_replace(Researcher, "not_recorded", "-1"),
         CaptureSite = str_replace(CaptureSite, "site_", ""),
         CaptureSite = str_replace(CaptureSite, "not_recorded", "-1"),
         Fisher = str_replace(Fisher, "fisher_", ""),
         Fisher = str_replace(Fisher, "not_recorded", "-1"),
         LandingSite = str_replace(LandingSite, "site_", ""),
         LandingSite = str_replace(LandingSite, "not_recorded", "-1"),
         ReleaseSite = str_replace(ReleaseSite, "site_", ""),
         ReleaseSite = str_replace(ReleaseSite, "not_recorded", "-1"),
         CaptureMethod = str_replace(CaptureMethod, "not_recorded", "-1"),
         Species = str_replace(Species, "species_", ""),
         Species = str_replace(Species, "not_recorded", "-1"),
         Tag_1_01 = str_replace(Tag_1, "(NotTagged_\\d+)", ""),
         Tag_1_01 = str_replace(Tag_1_01, "([^\\d]+)", ""),
         Tag_1_01 = ifelse(str_length(Tag_1_01) > 0, Tag_1_01, "-2"),
         Tag_1_02 = str_replace(Tag_1, "([\\d_]+)", ""),
         Tag_2_01 = str_replace(Tag_2, "None", "-1"),
         Tag_2_01 = str_replace(Tag_2_01, "([^\\-\\d]+)", ""),
         Tag_2_02 = str_replace(Tag_2, "([\\d_]+)", ""),
         SpecialRemarksbool = ifelse(is.na(SpecialRemarks), 1, 0),
         TurtleCharacteristicsbool = ifelse(is.na(TurtleCharacteristics), 1, 0),
         TurtleCharacteristics = replace_na(TurtleCharacteristics, ""),
         Release_Admiss_Notes = replace_na(Release_Admiss_Notes, ""),
         tchar_len = str_length(TurtleCharacteristics),
         tchar_uppercase_letters = str_count(TurtleCharacteristics, "[A-Z]"),
         tchar_extra_spaces = ifelse(str_detect(TurtleCharacteristics, "  "), 1, 0),
         tchar_wrong_char = str_replace_all(TurtleCharacteristics, "([:graph:]|[:blank:])", ""),
         tchar_wrong_char = ifelse(tchar_wrong_char != "", 1, 0),
         relan_len = str_length(Release_Admiss_Notes),
         relan_uppercase_letters = str_count(Release_Admiss_Notes, "[A-Z]"), 
         ForagingGround = case_when(ForagingGround == "creek" ~ 1,
                                    ForagingGround == "ocean" ~ 2,
                                    ForagingGround == "not_recorded" ~ 3)) %>%
  mutate_at(vars(Tag_1_02, Tag_2_02),
            list(~as_factor(.))) %>%
  mutate_at(vars(Tag_1_01, Tag_2_01, Tag_1_02, Tag_2_02),
            list(~as.numeric(.))) %>%
  replace_na(list(Sex = -2, Researcher = -2, CaptureSite = -2,
                  Fisher = -2, LandingSite = -2, ReleaseSite = -2,
                  Tag_1_01 = -2, Tag_1_02 = -2, Tag_2_01 = -2,
                  Tag_2_02 = -2, Status = -2, ForagingGround = -2,
                  CaptureMethod = -2, Weight_Kg = -2,
                  CCL_cm = -2, CCW_cm = -2)) %>%
  rename(CCL_cm_01 = CCL_cm,
         CCW_cm_01 = CCW_cm,
         Researcher_01 = Researcher,
         CaptureSite_01 = CaptureSite,
         ForagingGround_01 = ForagingGround,
         CaptureMethod_01 = CaptureMethod,
         Fisher_01 = Fisher,
         LandingSite_01 = LandingSite,
         Species_01 = Species,
         ReleaseSite_01 = ReleaseSite,
         Weight_Kg_01 = Weight_Kg,
         Status_01 = Status,
         Sex_01 = Sex) %>%
  mutate_at(vars(Researcher_01, CaptureSite_01, CaptureMethod_01, 
                 Fisher_01, LandingSite_01, ReleaseSite_01,
                 Species_01, Status_01, Sex_01, Tag_1_02),
            list(~as.numeric(as.factor(.)))) %>%
  select(Date_Caught_year,
         Date_Caught_day,
         Date_Caught_wday,
         Date_Caught_jul,
         Date_Release_year,
         tchar_len,
         tchar_extra_spaces,
         tchar_wrong_char,
         Researcher_01,
         CCL_cm_01,
         CCW_cm_01,
         Weight_Kg_01,
         Researcher_01,
         CaptureSite_01,
         ForagingGround_01,
         CaptureMethod_01,
         Fisher_01,
         Status_01,
         LandingSite_01,
         Species_01,
         ReleaseSite_01,
         SpecialRemarksbool,
         Tag_1_01, Tag_2_01,
         Sex_01,
         Tag_1_02,
         Tag_2_02,
         comp_cap_land, 
         comp_cap_release, 
         comp_release_land,
         Rescue_ID) %>%
  mutate(Tag_2_minus_1 = Tag_2_01 - Tag_1_01,
         Tag_2_minus_1 = ifelse(Tag_2_minus_1 == 1, 1, 0),
         Tag_1_01bool = ifelse(Tag_1_01 > 0, 1, 0),
         Tag_2_01bool = ifelse(Tag_2_01 > 0, 1, 0),
         date_release_bool = ifelse(is.na(Date_Release_year), 0, 1)) %>%
  mutate_if(is.logical, as.integer) %>%
  mutate_at(vars(ReleaseSite_01, LandingSite_01, Date_Caught_year, ForagingGround_01,
                 SpecialRemarksbool),
            list(~as.integer(.))) %>%
  select(-Date_Release_year, -Rescue_ID, -Tag_1_01, -Tag_2_02)

# Hyperparamter tuning
target_names <- colnames(target)
for (i in c(2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25)){
  cur_target <- target_names[i]
  # The Tag_3, T_Number, PCVNumber, Expenditure columns had no error.
  if (cur_target %in% c("Tag_3", "T_Number", "PCVNumber", "Expenditure")){
    tibble(response = rep(FALSE, 1362)) %>% write_csv(glue("prediction_1_{cur_target}.csv"))
  } else {
    train_01 <- data %>% slice(1:4292)
    train <- target %>% as_tibble() %>% select(starts_with(cur_target)) %>% cbind(train_01)
    train <- removeConstantFeatures(train)
    
    lrn = makeLearner("classif.xgboost", predict.type = "prob")
    tsk <- makeClassifTask(data = train, target = cur_target)
    
    ps <- makeParamSet(
      makeNumericParam(id = "nrounds", lower = log2(10/10), upper = log2(4000/10), trafo = function(x) round(2^x * 10), default = log2(10/10)),
      makeIntegerParam(id = "max_depth", default = 6L, lower = 1L, upper = 12L),
      makeNumericParam(id = "eta", lower = 0.05, upper = 0.05),
      makeNumericParam(id = "gamma", default = 0, lower = 0, upper = 10),
      makeNumericParam(id = "colsample_bytree", default = 0.5, lower = 0.3, upper = 0.7),
      makeNumericParam(id = "min_child_weight", default = 1, lower = 0, upper = 20),
      makeNumericParam(id = "subsample", default = 1, lower = 0.25, upper = 1),
      makeDiscreteParam(id = "nthread", values = 4)
    )
    
    set.seed(1)
    rdesc = makeResampleDesc("CV", iters = 5, stratify.cols = "Date_Caught_year")
    tc = makeTuneControlMBO(budget = 200, tune.threshold = TRUE)
    tr <- tuneParams(learner = lrn, task = tsk, resampling = rdesc, measures = acc, 
                     par.set = ps, control = tc)
    
    as_tibble(tr$x) %>% write_csv(path = glue("models/opt_params_{cur_target}.csv"))
    lrn <- setHyperPars(lrn, par.vals=tr$x)
    opt_value <- tr$y %>% as.vector()
    mdl <- train(lrn, tsk)
    
    test_data_x <- data %>% slice(4293:5654)
    prd <- predict(mdl, newdata = test_data_x)
    prd$data %>% write_csv(path = glue("models/prediction_{opt_value}_{cur_target}.csv"))
    mdl %>% save(file = glue("models/prediction_{opt_value}_{cur_target}.Rdata"))
  }
}