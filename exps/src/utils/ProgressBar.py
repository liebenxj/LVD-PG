from copy import deepcopy
import time
import numpy as np


class ProgressBar():
    def __init__(self, total_epochs, total_batches, statistics_name, cumulate_statistics = False):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.statistics_name = deepcopy(statistics_name)
        self.cumulate_statistics = cumulate_statistics

        self.cum_stats_vals = np.zeros([len(self.statistics_name)])
        
        self.batch_display_digits = len(str(self.total_batches))
        
        self.current_epoch = 0
        self.current_batch = 0
        
        self.per_batch_speed = 0.0
        self.last_time_s = 0.0
        self.epoch_start_time_s = 0.0
        
        self.max_str_length = 0
        
    def reset(self):
        self.current_epoch = 0
        self.current_batch = 0
        
        self.per_batch_speed = 0.0
        self.last_time_s = 0.0
        self.epoch_start_time_s = 0.0
        
        self.max_str_length = 0
        
    def new_epoch_begin(self):
        self.current_epoch += 1
        self.current_batch = 0
        
        self.per_batch_speed = 0.0
        
        if self.total_epochs > 1:
            print("Epoch {}/{}".format(self.current_epoch, self.total_epochs))
        
        self.last_time_s = time.time() / 1e9
        self.epoch_start_time_s = self.last_time_s

        self.cum_stats_vals *= 0.0
        
        self.print_line()

    def set_epoch_id(self, epoch_id):
        self.current_epoch = epoch_id
        
    def new_batch_done(self, statistics_val = None, n = 1):
        self.current_batch += n
        
        if self.current_batch > self.total_batches:
            self.current_batch = self.total_batches
        
        curr_time_s = time.time() / 1e9
        
        if (curr_time_s - self.last_time_s) / n * 10 < self.per_batch_speed:
            self.per_batch_speed = (curr_time_s - self.last_time_s) / n
        else:
            self.per_batch_speed = self.per_batch_speed * (self.current_batch - n) / self.current_batch + \
                (curr_time_s - self.last_time_s) / self.current_batch
        
        self.last_time_s = curr_time_s

        if self.cumulate_statistics:
            for i in range(len(self.statistics_name)):
                self.cum_stats_vals[i] += statistics_val[i]
            self.print_line(statistics_val = self.cum_stats_vals / self.current_batch)
        else:
            self.print_line(statistics_val = statistics_val)
        
        if self.current_batch == self.total_batches:
            # print("")
            pass
        
    def epoch_ends(self, statistics_val = None):
        if self.cumulate_statistics:
            self.print_line(statistics_val = self.cum_stats_vals / self.current_batch)
        else:
            self.print_line(statistics_val = statistics_val)
        print("")
            
    def print_line(self, statistics_val = None):
        progress_20 = (self.current_batch * 20) // self.total_batches
        
        string = "\r{:0" + str(self.batch_display_digits) + "d}/{:0" + str(self.batch_display_digits) + "d} ["
        
        string = string.format(self.current_batch, self.total_batches)
        
        if progress_20 == 0:
            string += " " * 20
        elif progress_20 <= 19:
            string += "=" * (progress_20 - 1) + ">" + " " * (20 - progress_20)
        else:
            string += "=" * 20
            
        string += "] - "
        
        if self.current_batch < self.total_batches:
            # Display remaining time
            remaining_time_s = self.per_batch_speed * (self.total_batches - self.current_batch)
        else:
            # Display total time
            remaining_time_s = self.last_time_s - self.epoch_start_time_s
        
        if remaining_time_s > 60 * 60:
            remaining_hours = int(remaining_time_s // 360)
            remaining_mins = int((remaining_time_s - remaining_hours * 360) // 60)
            
            string += str(remaining_hours) + "h" + str(remaining_mins) + "m"
            
        elif remaining_time_s > 60:
            remaining_mins = int(remaining_time_s // 60)
            remaining_secs = int(remaining_time_s - remaining_mins * 60)
            
            string += str(remaining_mins) + "m" + str(remaining_secs) + "s"
            
        else:
            remaining_secs = int(remaining_time_s)
            
            string += str(remaining_secs) + "s"
            
        string += " "
        
        if self.per_batch_speed > 1.0:
            string += str(int(self.per_batch_speed)) + "s/step"
        elif self.per_batch_speed > 1e-3:
            string += str(int(self.per_batch_speed * 1e3)) + "ms/step"
        elif self.per_batch_speed > 1e-6:
            string += str(int(self.per_batch_speed * 1e6)) + "us/step"
        else:
            string += "0s/step"
            
        if statistics_val is not None:
            for name, val in zip(self.statistics_name, statistics_val):
                string += " - " + name + ": " + str(round(val, 4))
                
        if len(string) > self.max_str_length:
            self.max_str_length = len(string)
        else:
            string += " " * (self.max_str_length - len(string))
                
        print(string, end = "")
        
    def print_validation_results(self, statistics_val = None, dataset_name = ""):
        if statistics_val is None:
            return
        
        string = "Validation set"
        if dataset_name != "":
            string += " [{}]".format(dataset_name)
        
        if statistics_val is not None:
            for name, val in zip(self.statistics_name, statistics_val):
                string += " - " + name + ": " + str(round(val, 4))
                
        print(string)