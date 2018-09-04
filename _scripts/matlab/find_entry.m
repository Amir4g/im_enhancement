function [entry , char_count_end] = find_entry(category_record , char_count_start)
flag = false;
entry = [];
char_count = char_count_start;
while flag == false
    if category_record(char_count) ~= ','
        entry = [entry,category_record(char_count)];
        char_count = char_count + 1;
    else
        flag = true;
    end
end
char_count_end = char_count + 1;