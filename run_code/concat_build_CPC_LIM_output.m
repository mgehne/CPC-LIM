function out = concat_build_CPC_LIM_output()

% This is a post-processing script that concats data that was created using the Python CPC-LIM package, and then writes it to a single netCDF file for scoring.
% ------------------------------------------------------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------------------------------------
% Set main directory and function paths
mainDir = ['/Projects/jalbers_process/CPC_LIM/'];
envPath=getenv('LD_LIBRARY_PATH');
setenv('LD_LIBRARY_PATH',['/usr/lib:/usr/lib64:',envPath])
% ------------------------------------------------------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------------------------------------
saveData = 'yes';

% Set path of data location and variable to read in and 
% variable write descriptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
limDir = ['coastal_LIM_v1.0_2.7.2025/'];
expName = ['full_NH_ssh'];
dataType = ['ZOS'];
    varDescription = strcat(dataType,'_anom');
    varUnits = 'meters';
    attrDataType = 'CPC dcoastal LIM v1.0'; 
    writeDir = [mainDir,limDir,'/processed_data'];
foldBounds = {[1993:1995],[1996:1998],[1999:2001],[2002:2004],[2005:2007],[2008:2010],[2011:2013],[2014:2016],[2017:2018],[2019:2020]};
offset = 'no';  % Set whether an climo offset was used to create the LIM data


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
readDir = [mainDir,limDir,'Images_',expName,'_hindcast_fold_'];
logIndx = 1;

for tf=1:length(foldBounds) % cycle through the number of cross-validation hindcast folds
    currentFold = foldBounds{tf};
    
    for ty=1:length(currentFold)  % cycle through the number of years in current hindcast fold
        loy = day(datetime(currentFold(ty),12,31),'dayofyear');
        
        for td=1:loy  % Cycle through the days in current fold year
            % Create current text string for path name and file name
            if( td==1 )
                currentDay = datetime(currentFold(ty),1,1);
            else
                currentDay = currentDay + days(1);
            end
            yearRead = num2str(year(currentDay),'%02.f');
            monthRead = num2str(month(currentDay),'%02.f');
            dayRead = num2str(day(currentDay),'%02.f');
            dateString = strcat(yearRead,monthRead,dayRead);
            
            if( strcmp(offset,'no')==1 )
                readFile = [readDir,num2str(tf),'/',dateString,'/no_offset/',dataType,'/',dataType,'.',dateString,'.nc'];
            else
                error('no set up to process data that was created with a climo offset')
            end
            
            % Check to see if file exists (needed because depending on the time average window used to create the LIM, the first few days of a 
            % hindcast period may be empty. If file does not exist put date into log file.
            if exist(readFile, 'file')
                if( exist('dataOut')==1 )
                    dataIn = ncread(readFile,strcat(dataType,'_anom'));
                    timeIn = ncread(readFile,'time');
                    
                    dataOut = cat(4,dataOut,dataIn);
                    timeOut = cat(1,timeOut,timeIn);
                else
                    lon = ncread(readFile,'lon');
                    lat = ncread(readFile,'lat');
                    baseDate = ncreadatt(readFile,'time','units');
                    lead_time = ncread(readFile,'lead_time');
                    
                    timeOut = ncread(readFile,'time');
                    dataOut(:,:,:,1) = ncread(readFile,strcat(dataType,'_anom'));
                end
            else
              % File does not exist, write date to log file
              display(['Warning: file does not exist: ', dateString])
              dne_dates{logIndx} = dateString;
              logIndx = logIndx + 1;
            end  
            clearvars dataIn yearRead monthRead dayRead readFile dateString timeIn
        end
    end
    display(['finished with fold ',num2str(tf),' of ',num2str(length(foldBounds))])
end
% Make sure time is a single precision for writing as classic netCDF
timeOut = single(timeOut);




if( strcmp(saveData,'yes')==1 )
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Write data to netCDF file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    display('writing data to netCDF')

    % Define variable schema
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Define output NetCDF file schema 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    data.Name='/';
    data.Format='classic';
    % Get file size
    dataSize=[size(dataOut,1),size(dataOut,2),size(dataOut,3),size(dataOut,4)];

    % Define dimensions of main file (global attributes)
    data.Dimensions(1).Name='longitude';
    data.Dimensions(1).Length=dataSize(1);
    data.Dimensions(2).Name='latitude';
    data.Dimensions(2).Length=dataSize(2);
    data.Dimensions(3).Name='lead_time';
    data.Dimensions(3).Length=dataSize(3);
    data.Dimensions(4).Name='time';
    data.Dimensions(4).Length=Inf;
    % Define name and dimensions of main variable
    data.Variables(1).Name=dataType;
    data.Variables(1).Dimensions(1)=data.Dimensions(1);
    data.Variables(1).Dimensions(2)=data.Dimensions(2);
    data.Variables(1).Dimensions(3)=data.Dimensions(3);
    data.Variables(1).Dimensions(4)=data.Dimensions(4);
    data.Variables(1).Datatype = 'double';
    % Define names and dimensions of secondary variables
    data.Variables(2).Name='longitude';
    data.Variables(2).Dimensions(1)=data.Dimensions(1);
    data.Variables(2).Datatype = 'single';
    data.Variables(3).Name='latitude';
    data.Variables(3).Dimensions(1)=data.Dimensions(2);
    data.Variables(3).Datatype = 'single';
    data.Variables(4).Name='lead_time';
    data.Variables(4).Dimensions(1)=data.Dimensions(3);
    data.Variables(4).Datatype = 'single';
    data.Variables(5).Name='time';
    data.Variables(5).Dimensions(1)=data.Dimensions(4);
    data.Variables(5).Datatype = 'single';
    % Define attributes of main file
    attrTitle=varDescription;
    data.Attributes(1).Name='Dataset';
    data.Attributes(1).Value =attrDataType;
    data.Attributes(2).Name='long_name';
    data.Attributes(2).Value =attrTitle;

    % Define attributes of variables
    data.Variables(1).Attributes(1).Name='long_name';
    data.Variables(1).Attributes(1).Value=attrTitle;
    data.Variables(1).Attributes(2).Name='units';
    data.Variables(1).Attributes(2).Value=varUnits;

    data.Variables(2).Attributes(1).Name='long_name';
    data.Variables(2).Attributes(1).Value='longitude';
    data.Variables(2).Attributes(2).Name='units';
    data.Variables(2).Attributes(2).Value='degrees_east';      

    data.Variables(3).Attributes(1).Name='long_name';
    data.Variables(3).Attributes(1).Value='latitude';
    data.Variables(3).Attributes(2).Name='units';
    data.Variables(3).Attributes(2).Value='degrees_north';

    data.Variables(4).Attributes(1).Name='long_name';
    data.Variables(4).Attributes(1).Value='lead_time';
    data.Variables(4).Attributes(2).Name='units';
    data.Variables(4).Attributes(2).Value='days';
    
    data.Variables(5).Attributes(1).Name='units';
    data.Variables(5).Attributes(1).Value=baseDate;
    data.Variables(5).Attributes(2).Name='calendar';
    data.Variables(5).Attributes(2).Value='proleptic_gregorian';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Write data out
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~exist(writeDir, 'dir')
       mkdir(writeDir)
    end
    writeName = strcat(varDescription,'_',num2str(foldBounds{1}(1)),'to',num2str(foldBounds{end}(end)),'.nc');
    
    ncwriteschema([writeDir,'/',writeName],data);
    ncwrite([writeDir,'/',writeName],dataType,dataOut);
    ncwrite([writeDir,'/',writeName],'longitude',lon);
    ncwrite([writeDir,'/',writeName],'latitude',lat);
    ncwrite([writeDir,'/',writeName],'lead_time',lead_time);
    ncwrite([writeDir,'/',writeName],'time',timeOut);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Write log file with dates that did not exist
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    display('writing missing dates to log file')
    t_write = table(dne_dates');
    t_write.Properties.VariableNames = {'Missing dates'};
    writetable(t_write,[writeDir,'/','log_file_',varDescription,'_missing_dates.txt'])

end















end