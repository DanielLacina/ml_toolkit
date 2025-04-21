use crate::dataframe::{DataFrame, DataTypeValue};
use std::collections::HashMap;
use std::collections::HashSet;

pub struct StratifiedShuffleSplit {
    test_size: f32,
    stratified_by: Vec<(String, usize)>,
}

impl StratifiedShuffleSplit {
    pub fn new(test_size: f32, stratified_by: &Vec<(String, usize)>) -> Self {
        if test_size < 0.0  || test_size > 1.0 {
            panic!("test size must be a percentage");
        }
        if stratified_by.len() == 0 {
            panic!("must provide at least one column to stratify the samples by"); 
        }
        if stratified_by.iter().any(|(_, num_bins)| *num_bins == 0) {
            panic!("cant divide data into 0 bins");
        } 
        Self {
            test_size,
            stratified_by: stratified_by.clone()
        }
    }

    pub fn split(&self, df: &DataFrame) -> (Vec<usize>, Vec<usize>) {
        let mut test_indices: Vec<usize> = Vec::new();
        let mut train_indices: Vec<usize> = Vec::new();
        let mut bin_permutations = Vec::new();
        for (_, num_bins) in self.stratified_by.iter() {
            if bin_permutations.len() == 0 { 
                for i in (0..*num_bins)  {
                    bin_permutations.push(vec![i as u32]);
                }
            } else {
                let bin_permutations_copy = bin_permutations.clone();
                bin_permutations = Vec::new();
                for i in (0..*num_bins) {
                    let start = if i >= bin_permutations_copy.len() {
                        0     
                    } else {
                        i
                    };
                    for bin in bin_permutations_copy[start..].iter() {
                        let mut bin = bin.clone();
                        bin.push(i as u32); 
                        bin_permutations.push(bin);
                    }
                }
            }
        }
        let mut bin_permutations_hashmap = HashMap::new();
        for bin_nums in bin_permutations.into_iter() {
            let mut bin_nums = bin_nums; 
            bin_nums.sort();
            bin_permutations_hashmap.insert(bin_nums, Vec::new());
        } 
        let mut all_bins = Vec::new(); 
        for (column_name, num_bins) in self.stratified_by.iter() {
            let bins = df.bins(&column_name, *num_bins);
            if all_bins.len() == 0 {
                for (id, _, bin_num) in bins.iter() {
                    let id = match id {
                        DataTypeValue::Id(inner) => {
                            *inner
                        }, 
                        _ => panic!("datatype must be id")
                    };
                    all_bins.push((id.clone(), vec![*bin_num])); 
                }
            } else {
                for (i, (_, bin_nums)) in all_bins.iter_mut().enumerate() {
                    let (_, _, cur_bin_num) = bins[i]; 
                    bin_nums.push(cur_bin_num);
                }
            }
         } 
         let mut distinct_bin_nums = HashSet::new();
         for (id, bin_nums) in all_bins.into_iter() {
            let mut bin_nums = bin_nums;
            bin_nums.sort();
            distinct_bin_nums.insert(bin_nums.clone());
            let bin_ids = bin_permutations_hashmap.get_mut(&bin_nums).unwrap();
            bin_ids.push(id);
         }
         let mut distinct_bin_nums: Vec<Vec<u32>> = distinct_bin_nums.into_iter().collect();
         distinct_bin_nums.sort();
         for bin_nums in distinct_bin_nums.iter() {
            let ids = bin_permutations_hashmap.get(bin_nums).unwrap();
            let divisor = (1.0/self.test_size) as usize;  
            for id in ids {
                if *id % divisor == 0 {
                   test_indices.push(*id)
                }  
                else {
                    train_indices.push(*id);
                }
            }
         }
         return (train_indices, test_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::csv::df_from_csv; 
    use std::collections::HashSet;

   #[test]
   fn test_simple_stratified_shuffle_split() {
      let filename = "housing.csv";
      let row_limit = 100;
      let df = df_from_csv(filename, Some(row_limit));
      let test_size = 0.2; 
      let stratified_by = vec![("median_income".to_string(), 5)];
      let stratified_shuffle_split = StratifiedShuffleSplit::new(test_size, &stratified_by);
      let (train_indices, test_indices) = stratified_shuffle_split.split(&df);
      assert!(train_indices.len() == (df.len() as f32 * (1.0 - test_size)) as usize);
      assert!(test_indices.len() == (df.len()  as f32 * test_size) as usize);
      let mut id_hashset = HashSet::new();
      let mut all_indices = Vec::new();
      all_indices.extend(train_indices); 
      all_indices.extend(test_indices);
      assert!(all_indices.iter().all(|index| {
        if id_hashset.contains(index) {
            false
        } else {
            id_hashset.insert(index.clone());
            true
        }
      }));
   }

      #[test]
   fn test_multiple_stratified_shuffle_split() {
      let filename = "housing.csv";
      let row_limit = 100;
      let df = df_from_csv(filename, Some(row_limit));
      let test_size = 0.2; 
      let stratified_by = vec![("median_income".to_string(), 5), ("households".to_string(), 2)];
      let stratified_shuffle_split = StratifiedShuffleSplit::new(test_size, &stratified_by);
      let (train_indices, test_indices) = stratified_shuffle_split.split(&df);
      assert!(train_indices.len() == (df.len() as f32 * (1.0 - test_size)) as usize);
      assert!(test_indices.len() == (df.len()  as f32 * test_size) as usize);
      let mut id_hashset = HashSet::new();
      let mut all_indices = Vec::new();
      all_indices.extend(train_indices); 
      all_indices.extend(test_indices);
      assert!(all_indices.iter().all(|index| {
        if id_hashset.contains(index) {
            false
        } else {
            id_hashset.insert(index.clone());
            true
        }
      }));
   }
}
