function out_categorical = convert_to_one_shot(name , class_type)
switch name
    case 'day'
        out_categorical = [1,0,0,0];
    case 'night'
        out_categorical = [0,1,0,0];
    case 'dusk'
        out_categorical = [0,0,1,0];
    case 'outdoor'
        out_categorical = [1,0,0];
    case 'indoor'
        out_categorical = [0,1,0];
    case 'sun_sky'
        out_categorical = [1,0,0,0];
    case 'artificial'
        out_categorical = [0,1,0,0];
    case 'mixed'
        out_categorical = [0,0,1,0];
    case 'people'
        out_categorical = [1,0,0,0,0,0];
    case 'man_made'
        out_categorical = [0,1,0,0,0,0];
    case 'nature'
        out_categorical = [0,0,1,0,0,0];
    case 'animals'
        out_categorical = [0,0,0,1,0,0];
        case 'abstract'
        out_categorical = [0,0,0,0,1,0];
    case 'unknown'
        switch class_type
            case 'type'
                out_categorical = [0,0,1];
            case 'time'
                out_categorical = [0,0,0,1];
            case 'light'
                out_categorical = [0,0,0,1];
            case 'subject'
                out_categorical = [0,0,0,0,0,1];
        end
    case 'None'
        switch class_type
            case 'time'
                out_categorical = [0,0,0,1];
            case 'type'
                out_categorical = [0,0,1];
            case 'light'
                out_categorical = [0,0,0,1];
            case 'subject'
                out_categorical = [0,0,0,0,0,1];
        end
        
    otherwise
        error('undefined class label')
        
        
end