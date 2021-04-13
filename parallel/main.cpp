#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <pthread.h>

using namespace std;

#define TRAIN_FILE_NAME_PREFIX "train_"
#define CSV_FILE ".csv"
#define WEIGHTS_FILE_NAME "weights.csv"
#define DIVIDER '/'
#define DATASET_IND 1
#define SPACE ' '
#define COMMA ','
#define PERCANTAGE '%'
#define FEATURE_CNT 20
#define MAX_THREAD_NUMBER 30
#define NORMALIZATION_THREAD_NUMBER 20
#define CALCULATION_THREAD_NUMBER 20
#define DATA_FILE_THREAD_NUMBER 4

const double INF = 100000000;

typedef vector < double > Feature_value_list;
typedef vector < double > Weight_list;
typedef int Mobile_type;
typedef double Price_class_bios;

struct Mobile_data{
	Feature_value_list feature;
	Mobile_type type;

	void add_feature(double feature_value){
		feature.push_back(feature_value);
	}
	int features_cnt(){
		return feature.size();
	}
};

struct Price_class{
	Weight_list weight;
	Price_class_bios bios;

	void add_feature_weight(double feature_weight){
		weight.push_back(feature_weight);
	}

	double calculate_class_score(Mobile_data cur_mobile_data){
		double result = 0;
		for(int feature_ind = 0 ; feature_ind < cur_mobile_data.features_cnt() ; feature_ind++){
			result += weight[feature_ind] * cur_mobile_data.feature[feature_ind];
		}
		return result + bios;
	}
};

typedef vector < Mobile_data > Mobiles_data_list;
typedef vector < Price_class > Price_class_list;

struct Calculation_data{
    Price_class_list* class_data;
    int tid;
};

struct Min_max_data{
    double min_val,max_val;
    int feature_ind;
    int tid;
};

//global variables for threads
pthread_mutex_t mobiles_data_mutex , correct_mutex;
Mobiles_data_list mobiles_data;
int correct = 0;
string dataset_path;
//end of thead shared variables

string replace_comma_with_space(string x){
	for(int i = 0 ; i < x.size() ; i++){
		if(x[i] == COMMA){
			x[i] = SPACE;
		}
	}
	return x;
}

string get_train_file_path(int tid){
	if(dataset_path[dataset_path.size() - 1] == DIVIDER)
		return dataset_path + TRAIN_FILE_NAME_PREFIX + to_string(tid) + CSV_FILE;
	return dataset_path + DIVIDER + TRAIN_FILE_NAME_PREFIX + to_string(tid) + CSV_FILE;
}

string get_weight_file_path(string dataset_path){
	if(dataset_path[dataset_path.size() - 1] == DIVIDER)
		return dataset_path + WEIGHTS_FILE_NAME;
	return dataset_path + DIVIDER + WEIGHTS_FILE_NAME;
}

int find_number_of_features(ifstream& train_file){
	int result = 0;
	string name_line;
	getline(train_file , name_line);
	stringstream ss(replace_comma_with_space(name_line));

	string cur_feature_name;
	while(ss >> cur_feature_name)
		result++;
	return result - 1;
}

void read_and_store_data_from_train_file(ifstream& train_file , Mobiles_data_list& mobiles_data){
	int features_cnt = find_number_of_features(train_file);
	string data_line;
	while(getline(train_file , data_line)){
		stringstream ss(replace_comma_with_space(data_line));
		Mobile_data new_mobile_data;

		double tmp_feature_value;
		for(int feature_ind = 0 ; feature_ind < features_cnt ; feature_ind++){
			ss >> tmp_feature_value;
			new_mobile_data.add_feature(tmp_feature_value);
		}
		ss >> new_mobile_data.type;

		mobiles_data.push_back(new_mobile_data);
	}
}

void* find_min_and_max(void* arg){
    Min_max_data* min_max_data_p = (Min_max_data*)(arg);

    int feature_ind = min_max_data_p -> feature_ind;
    int offset = (min_max_data_p -> tid);
    int step = mobiles_data.size() / DATA_FILE_THREAD_NUMBER;
    int start = offset * step;
    int end = start + step;

    if(offset == DATA_FILE_THREAD_NUMBER - 1)
        end = mobiles_data.size();
    
    for(int mobile_ind = start ; mobile_ind < end ; mobile_ind++){
        min_max_data_p -> min_val = min(min_max_data_p->min_val , mobiles_data[mobile_ind].feature[feature_ind]);
        min_max_data_p -> max_val = max(min_max_data_p->max_val , mobiles_data[mobile_ind].feature[feature_ind]);
    }

    pthread_exit(NULL);
}

void* normalize_mobiles_data(void* arg){
    int* feature_ind_p = (int*)(arg);
    int feature_ind = *(feature_ind_p);

    pthread_t thread_id[DATA_FILE_THREAD_NUMBER];
    Min_max_data min_max_thread_data[DATA_FILE_THREAD_NUMBER];

    for(int tid = 0 ; tid < DATA_FILE_THREAD_NUMBER ; tid++){
        min_max_thread_data[tid].min_val = INF;
        min_max_thread_data[tid].max_val = 0;
        min_max_thread_data[tid].feature_ind = feature_ind;
        min_max_thread_data[tid].tid = tid;

        pthread_create(&thread_id[tid] , NULL , find_min_and_max , (void*)&min_max_thread_data[tid]);
    }

    double min_val = INF , max_val = 0;
    for(int tid = 0 ; tid < DATA_FILE_THREAD_NUMBER ; tid++){
        pthread_join(thread_id[tid] , NULL);
        min_val = min(min_val , min_max_thread_data[tid].min_val);
        max_val = max(max_val , min_max_thread_data[tid].max_val);
    }

    pthread_mutex_lock(&mobiles_data_mutex);
	for(int mobile_ind = 0 ; mobile_ind < mobiles_data.size() ; mobile_ind++){
		Mobile_data* cur_mobile = &mobiles_data[mobile_ind];
		cur_mobile->feature[feature_ind] -= min_val;
		cur_mobile->feature[feature_ind] /= (max_val - min_val);
	}
    pthread_mutex_unlock(&mobiles_data_mutex);
    pthread_exit(NULL);
}

void read_and_store_data_from_weight_file(ifstream& weight_file , Price_class_list& price_class_data){
	int feature_cnt = find_number_of_features(weight_file);
	string data_line;
	while(getline(weight_file , data_line)){
		stringstream ss(replace_comma_with_space(data_line));
		Price_class new_price_class;

		double tmp_weight;
		for(int feature_ind = 0 ; feature_ind < feature_cnt ; feature_ind++){
			ss >> tmp_weight;
			new_price_class.add_feature_weight(tmp_weight);
		}
		ss >> new_price_class.bios;

		price_class_data.push_back(new_price_class);
	}
}

void* extract_data_from_trainfile(void* arg){
    int* tid = (int*)(arg);
	string train_file_path = get_train_file_path(*tid);
	ifstream train_file(train_file_path);
    Mobiles_data_list cur_thread_mobiles_data;
	read_and_store_data_from_train_file(train_file , cur_thread_mobiles_data);

    pthread_mutex_lock(&mobiles_data_mutex);
    //critical section
    for(int cur_thread_data_ind = 0 ; cur_thread_data_ind < cur_thread_mobiles_data.size() ; cur_thread_data_ind++){
        mobiles_data.push_back(cur_thread_mobiles_data[cur_thread_data_ind]);
    }
    pthread_mutex_unlock(&mobiles_data_mutex);

    train_file.close();
    pthread_exit(NULL);
}

void extract_weights_from_weight_file(string dataset_path , Price_class_list& price_class_data){
	string weight_file_path = get_weight_file_path(dataset_path);
	ifstream weight_file(weight_file_path);
	read_and_store_data_from_weight_file(weight_file , price_class_data);
	weight_file.close();
}

void* calculate_accuracy(void* arg){
    Calculation_data* calc_data_p = (Calculation_data*)(arg);
    Price_class_list price_class_data = *(calc_data_p->class_data); 
    
    int offset = (calc_data_p->tid);
    int step = mobiles_data.size() / CALCULATION_THREAD_NUMBER;
    int start = offset * step;
    int end = start + step;

    if(offset == CALCULATION_THREAD_NUMBER - 1)
        end = mobiles_data.size();
    
    int temp_cor = 0;
    for(int mobile_ind = start ; mobile_ind < end ; mobile_ind++){
		Mobile_data cur_mobile = mobiles_data[mobile_ind];
		int calculated_class = 0;
		double score = 0;
		for(int price_class_ind = 0 ; price_class_ind < price_class_data.size() ; price_class_ind++){
			Price_class cur_class = price_class_data[price_class_ind];
			double cur_class_score = cur_class.calculate_class_score(cur_mobile);
			calculated_class = (score < cur_class_score) ? price_class_ind : calculated_class;
			score = (score < cur_class_score) ? cur_class_score : score;
		}
		if(calculated_class == cur_mobile.type)
		    temp_cor += 1;        
    }
    pthread_mutex_lock(&correct_mutex);
    correct += temp_cor;
    pthread_mutex_unlock(&correct_mutex);

    pthread_exit(NULL);
}

int main(int argc , char* argv[]){
	clock_t start1 , end1 , start2 , end2 , start3 , end3;
	if(argc != 2){
		cout << "Input Error!\n" << endl;
		return 0;
	}
	Price_class_list price_class_data;

    pthread_t thread_id[MAX_THREAD_NUMBER];
    int tid_arr[MAX_THREAD_NUMBER];
    pthread_mutex_init(&mobiles_data_mutex , NULL);
    pthread_mutex_init(&correct_mutex , NULL);

    dataset_path = argv[DATASET_IND];

    for(int tid = 0 ; tid < DATA_FILE_THREAD_NUMBER ; tid++){
        tid_arr[tid] = tid;
        pthread_create(&thread_id[tid] , NULL , extract_data_from_trainfile , (void*)&tid_arr[tid]);
    }
    for(int tid = 0 ; tid < DATA_FILE_THREAD_NUMBER ; tid++)
        pthread_join(thread_id[tid] , NULL);


    for(int tid = 0 ; tid < NORMALIZATION_THREAD_NUMBER ; tid++){
        tid_arr[tid] = tid;
        pthread_create(&thread_id[tid] , NULL , normalize_mobiles_data , (void*)&tid_arr[tid]);
    }
    for(int tid = 0 ; tid < NORMALIZATION_THREAD_NUMBER ; tid++)
        pthread_join(thread_id[tid] , NULL);

	extract_weights_from_weight_file(dataset_path , price_class_data);

    Calculation_data thread_calc_data[MAX_THREAD_NUMBER];
    for(int tid = 0 ; tid < CALCULATION_THREAD_NUMBER ; tid++){
        thread_calc_data[tid].class_data = &price_class_data;
        thread_calc_data[tid].tid = tid;

        pthread_create(&thread_id[tid] , NULL , calculate_accuracy , (void*)&thread_calc_data[tid]);
    }
    for(int tid = 0 ; tid < CALCULATION_THREAD_NUMBER ; tid++)
        pthread_join(thread_id[tid] , NULL);

    double accuracy = double(correct) / mobiles_data.size();
	cout << fixed << setprecision(2) << "Accuracy: " << 100 * accuracy << PERCANTAGE << endl;
}