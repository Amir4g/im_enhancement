function [main_dir,photo_dir] = getpaths(user_name)

%user name = 'kamyab' , 'nazemi' , 'server'

if strcmp(user_name , 'kamyab')
    main_dir = 'E:\thesis_phd\Image_enhancement_MrNazemi\Dataset\fivek_dataset';
    photo_dir = 'E:\thesis_phd\Image_enhancement_MrNazemi\Dataset\fivek_dataset\raw_photos';
elseif strcmp(user_name , 'nazemi')
    % TO DO
elseif strcmp(user_name , 'server')
    % TO DO
end