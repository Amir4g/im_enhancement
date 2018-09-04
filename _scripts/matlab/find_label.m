function [label_categorical, label_string] = find_label(main_dir , img_name)

all_labels = importdata([main_dir , '\' , 'categories.txt']);
% label_categorical = zeros(length(all_labels) ,17 );
for i = 1 : length(all_labels)
    category_record = all_labels{i};
    char_count_start = 1;
    [name , char_count_end] = find_entry(category_record, char_count_start);
    
    if strcmp([name,'.dng'] , img_name)
        
        
        char_count_start = char_count_end;
        [name , char_count_end] = find_entry(category_record, char_count_start);
        photo_type_label = name;
        photo_type_categorical = convert_to_one_shot(photo_type_label , 'type');
        
        char_count_start = char_count_end;
        [name , char_count_end] = find_entry(category_record, char_count_start);
        capture_time_label = name;
        capture_time_categorical = convert_to_one_shot(capture_time_label,'time');
        
        
        char_count_start = char_count_end;
        [name , char_count_end] = find_entry(category_record, char_count_start);
        light_type_label = name;
        light_type_categorical = convert_to_one_shot(light_type_label, 'light');
        
        char_count_start = char_count_end;
        name = category_record(char_count_start:end);%find_entry(category_record, char_count_start);
        subject_label = name;
        subject_label_categorical = convert_to_one_shot(subject_label,'subject');
        
        label_categorical = [photo_type_categorical,...
            capture_time_categorical, ...
            light_type_categorical,...
            subject_label_categorical];
        
        label_string = {photo_type_label,capture_time_label, light_type_label ,  subject_label};
        
        break;
    end
end