mod algorithms;
mod dataframe;
mod inference;
mod linear_algebra;
mod pipeline;
mod sampling;
use algorithms::linear_regression::linear_regression::LinearRegression;
use dataframe::csv::df_from_csv;
use inference::inference::rmse;
// use pipeline::one_hot_encoder::df_one_hot_encoded;
use pipeline::pipeline::*;
use sampling::sampling::StratifiedShuffleSplit;

fn main() {
//     let filename = "housing.csv";
//     let df = df_from_csv(filename, None);
//     let stratified_shuffle_split =
//         StratifiedShuffleSplit::new(0.2, &vec![("median_income".to_string(), 10)]);
//     let (train_indices, test_indices) = stratified_shuffle_split.split(&df);
//     let train_set = df.get_rows_as_df(&train_indices);
//     let label_column = "median_house_value";
//     let labels = train_set.get_columns_as_df(&vec![label_column.to_string()]);
//     let mut features = train_set;
//     features.remove_column(label_column);
//     let features_pipeline = Pipeline::new(ImputerStrategy::Median, Scalar::Standard);
//     let column_names = vec!["ocean_proximity".to_string()];
//     let features = df_one_hot_encoded(&df, &column_names);
//     let inputs = features_pipeline.transform(&features);
//     let labels_pipeline = Pipeline::new(ImputerStrategy::Median, Scalar::None);
//     let labels = labels_pipeline.transform(&labels);
//     let mut lin_reg = LinearRegression::new(20.0);
//     lin_reg.fit(&inputs, &labels);
//     let labels = labels.into_iter().map(|v| v[0]).collect();
//     let predictions = lin_reg.predict(&inputs);
//     println!("{}", rmse(&predictions, &labels));
//     let test_set = df.get_rows_as_df(&test_indices);
//     let labels = test_set.get_columns_as_df(&vec![label_column.to_string()]);
//     let mut features = test_set;
//     features.remove_column(label_column);
//     let features = df_one_hot_encoded(&df, &column_names);
//     let inputs = features_pipeline.transform(&features);
//     let labels = labels_pipeline.transform(&labels);
//     lin_reg.predict(&inputs);
//     let predictions = lin_reg.predict(&inputs);
//     let labels = labels.into_iter().map(|v| v[0]).collect();
//     println!("{}", rmse(&predictions, &labels));
}
