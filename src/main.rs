mod dataframe;
mod inference;
mod algorithms;
mod linear_algebra;
mod pipeline;
mod sampling;
mod file_reader;


fn main() {
    // let filename = "housing.csv";
    // let mut features_df = df_from_csv(filename, None);
    // let labels_df = features_df.get_columns_as_df(&vec!["median_house_value".to_string()]);
    // println!("{:?}", labels_df.data());
    // features_df.remove_column("median_house_value");
    // let features_pipeline = Pipeline::new(
    //     StringEncoding::OneHot,
    //     ImputerStrategy::Median,
    //     Scalar::Standard,
    // );
    // let inputs = features_pipeline.transform(&features_df);
    // let labels_pipeline = Pipeline::new(
    //     StringEncoding::OneHot,
    //     ImputerStrategy::Median,
    //     Scalar::None,
    // );
    // let labels = labels_pipeline.transform(&labels_df);
    // let mut lin_reg = LinearRegression::new(10.0);
    // lin_reg.fit(&inputs, &labels);
    // let labels = labels.into_iter().map(|v| v[0]).collect();
    // let predictions = lin_reg.predict(&inputs);
    // println!("{}", rmse(&predictions, &labels));
}
