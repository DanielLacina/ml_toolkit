// use crate::dataframe::DataFrame;

// pub struct StratifiedShuffleSplit {
//     test_size: f32,
//     stratified_by: Vec<String>
// }

// impl StratifiedShuffleSplit {
//     pub fn new(test_size: f32) -> Self {
//         if test_size < 0.0  || test_size > 1.0 {
//             panic!("test size must be a percentage");
//         }
//         Self {
//             test_size
//         }
//     }

//     pub fn split(&self, df: &DataFrame) {
//         let columns = df.columns();
//     }
// }
