mod algorithms;
mod dataframe;
mod inference;
mod linear_algebra;
mod pipeline;
mod sampling;
use dataframe::csv::df_from_csv;
use sampling::sampling::StratifiedShuffleSplit;
use pipeline::pipeline::*;
use algorithms::linear_regression::linear_regression::LinearRegression;
use inference::inference::rmse;

fn main() {
    let filename = "housing.csv";
    let df = df_from_csv(filename, None);
    let stratified_shuffle_split = StratifiedShuffleSplit::new(0.2, &vec![("median_income".to_string(), 5)]);
    let (train_indices, test_indices) = stratified_shuffle_split.split(&df); 
    let train_set = df.get_rows_as_df(train_indices);
    let label_column = "median_house_value";
    let labels = train_set.get_columns_as_df(&vec![label_column.to_string()]);
    let mut features = train_set;
    features.remove_column(label_column);
    let features_pipeline = Pipeline::new(
        StringEncoding::OneHot,
        ImputerStrategy::Median,
        Scalar::Standard,
    );
    let inputs = features_pipeline.transform(&features);
    let labels_pipeline = Pipeline::new(
        StringEncoding::OneHot,
        ImputerStrategy::Median,
        Scalar::None,
    );
    let labels = labels_pipeline.transform(&labels);
    let mut lin_reg = LinearRegression::new(10.0);
    lin_reg.fit(&inputs, &labels);
    let labels = labels.into_iter().map(|v| v[0]).collect();
    let predictions = lin_reg.predict(&inputs);
    println!("{}", rmse(&predictions, &labels));
}
