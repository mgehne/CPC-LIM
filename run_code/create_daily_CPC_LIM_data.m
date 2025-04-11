function out = create_daily_CPC_LIM_data()
% This is a preprocessing script that creates data for use in the Python CPC-LIM package. The script reads in data, regrids it, and then saves it in the netCDF
% format that is required for the CPC-LIM. This includes adjusting the 'time since...' format and all other metadata to be congruent with the CPC-LIM package.
% ------------------------------------------------------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------------------------------------
% Set main directory and function paths
envPath=getenv('LD_LIBRARY_PATH');
setenv('LD_LIBRARY_PATH',['/usr/lib:/usr/lib64:',envPath])
% ------------------------------------------------------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------------------------------------------------------

% User defined flags
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
saveData = 'yes';    % Options: 'yes' or 'no'
% Set variables and write directory
readData = 'glorys';    % current options: glorys (sst and zos)
    readVar = 'zos';
    writeDir='/Projects/jalbers_process/CPC_LIM/coastal_LIM_v1.0_2.7.2025/jra55/';
% Define interpolation grid and regrid method
grid = 1;  % Output grid resolution    
    % Current regrid options include: 2D linear interpolation (intp2)
    intMethod='intp2';
    % Define new grid (NOTE: CPC LIM expects latitude grid starting at the North Pole, so the code below makes sure to enforce this before regridding)
    latsNew=flip([-90:grid:90]');
        flipData = 'no';  % FIXED (see NOTE above about ensuring correct grid orientation)
    lonsNew=[0:grid:360-grid]';
% Define number of years to be read in
years=[1995:1:2020];

% Fixed flags
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if( strcmp(readData,'glorys')==1 )
    % Define time attribute for GLORYS
    timeIncrement = 'hours';
    baseDateIn = 1950;
    if( strcmp(readVar,'zos')==1 )              
        % Define variable name convention of data to read in
        varRead = ['zos.daily.mean'];
        varWrite = ['zos_'];
        varNameIn='zos';
        
        % Define metadata and attributes for netCDF output file
        varNameOut='zos';
        attrDataType='MERCATOR GLORYS12V1 - regridded to 1-degree using 2D bilinear interpolation';
        varDescription='Sea surface height';
        varUnits='meters';

        % Open data (lons x lats x time)
        readDir = ['/Projects/GLORYS/Dailies/monolevel/'];
        lats=ncread([readDir,varRead,'.',num2str(years(1)),'.nc'],'latitude');
        lons=ncread([readDir,varRead,'.',num2str(years(1)),'.nc'],'longitude');
    end
    if( strcmp(readVar,'sst')==1 )              
        % Define variable name convention of data to read in
        varRead = ['sst.daily.mean'];
        varWrite = ['sst_'];
        varNameIn='thetao';
        
        % Define metadata and attributes for netCDF output file
        varNameOut='thetao';
        attrDataType='MERCATOR GLORYS12V1 - regridded to 1.25 degrees using 2D bilinear interpolation';
        varDescription='Temperature (standard_name: sea water potential temperature)';
        varUnits='degrees_C';

        % Open data (lons x lats x time)
        readDir = ['/Projects/GLORYS/Dailies/monolevel/'];
        lats=ncread([readDir,varRead,'.',num2str(years(1)),'.nc'],'latitude');
        lons=ncread([readDir,varRead,'.',num2str(years(1)),'.nc'],'longitude');
    end
end

for t=1:length(years)
    tic
    
    if( strcmp(intMethod,'intp2')==1 )
        % Read in original data
        data=squeeze(ncread([readDir,varRead,'.',num2str(years(t)),'.nc'],varNameIn));
       
        % CPC LIM latitude grid starts at the North Pole and extends southwards, so ensure this convention with the data by flipping in latitude if necessary
        if( lats(1)<lats(2) )
            flipData = 'yes';
            if( t==1 )
                lats = flip(lats);
            end
        end
        if( strcmp(flipData,'yes')==1 )
            data = flip(data,2);
        end
                        
        % Read in time data and attributes of original data
        clearvars time
        time=ncread([readDir,varRead,'.',num2str(years(t)),'.nc'],'time');
        
        if( strcmp(timeIncrement,'hours')==1 )
            
            % Create daily CPC-LIM formated date
            lim_baseDate_string = strcat(['days since ',num2str(years(t)),'-01-01']);    % Base date used by CPC-LIM 
            lim_baseDate = datetime(years(t),1,1);
            
            t1 = datetime(baseDateIn,1,1) + hours(time);    % Time of original dataset using its native base time    
            offset = t1 - lim_baseDate;
            time_out = lim_baseDate + offset;
            % Convert dates to start of day for ease of finding all 'hour times' that meet the current day criteria
            time_out = dateshift(time_out,'start','day');

            % Create output date
            hours_since_base = time_out - lim_baseDate;
            hours_since_base_out = hours(hours_since_base); % hours since baseDate 
            days_since_base_out = int64(hours_since_base_out/24);
            
            clearvars t1 offset time timeOut lim_baseDate hours_since_base_out
        end

        % Create interpolation grid
        [lonGrid,latGrid]=meshgrid(lons,lats);
        [lonGridNew,latGridNew]=meshgrid(lonsNew,latsNew);
        dataAvg=zeros(length(lonsNew),length(latsNew),size(data,3));

        % Conduct interpolation
        for n=1:size(data,3)
                dataAvg(:,:,n)=(interp2(lonGrid,latGrid,squeeze(data(:,:,n))',lonGridNew,latGridNew,'linear'))';
                
                plot_check = 'no';
                % Plot check
                if( strcmp(plot_check,'yes')==1 )
                    figure(1)
                    contourf(lonGridNew,latGridNew,squeeze(dataAvg(:,:,n))')
                    pause
                end
                               
                % Place data in output array
                if( n==1 )
                    dataOut=dataAvg(:,:,n);
                end
                if( n>1 )
                    dataOut=cat(3,dataOut,dataAvg(:,:,n));
                end
        end
        clearvars dataAvg data   
    end    

            
    if( strcmp(saveData,'yes')==1 )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Write data to netCDF file
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Define variable schema
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Define output NetCDF file schema 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data.Name='/';
        data.Format='netcdf4';
        % Get file size
        dataSize=[size(dataOut,1),size(dataOut,2),size(dataOut,3)];

        % Define dimensions of main file (global attributes)
        data.Dimensions(1).Name='longitude';
        data.Dimensions(1).Length=dataSize(1);
        data.Dimensions(2).Name='latitude';
        data.Dimensions(2).Length=dataSize(2);
        data.Dimensions(3).Name='time';
        data.Dimensions(3).Length=Inf;
        % Define name and dimensions of main variable
        data.Variables(1).Name=varNameOut;
        data.Variables(1).Dimensions(1)=data.Dimensions(1);
        data.Variables(1).Dimensions(2)=data.Dimensions(2);
        data.Variables(1).Dimensions(3)=data.Dimensions(3);
        data.Variables(1).Datatype = 'single';
        % Define names and dimensions of secondary variables
        data.Variables(2).Name='longitude';
        data.Variables(2).Dimensions(1)=data.Dimensions(1);
        data.Variables(2).Datatype = 'single';
        data.Variables(3).Name='latitude';
        data.Variables(3).Dimensions(1)=data.Dimensions(2);
        data.Variables(3).Datatype = 'single';
        data.Variables(4).Name='time';
        data.Variables(4).Dimensions(1)=data.Dimensions(3);
        data.Variables(4).Datatype = 'int64';
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
                
        data.Variables(4).Attributes(1).Name='units';
        data.Variables(4).Attributes(1).Value=lim_baseDate_string;
        data.Variables(4).Attributes(2).Name='calendar';
        data.Variables(4).Attributes(2).Value='proleptic_gregorian';

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Write data out
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ncwriteschema([writeDir,'/',num2str(years(t)),'/',varWrite,num2str(years(t)),'.nc'],data);
        ncwrite([writeDir,'/',num2str(years(t)),'/',varWrite,num2str(years(t)),'.nc'],varNameOut,dataOut);
        ncwrite([writeDir,'/',num2str(years(t)),'/',varWrite,num2str(years(t)),'.nc'],'longitude',lonsNew);
        ncwrite([writeDir,'/',num2str(years(t)),'/',varWrite,num2str(years(t)),'.nc'],'latitude',latsNew);
        ncwrite([writeDir,'/',num2str(years(t)),'/',varWrite,num2str(years(t)),'.nc'],'time',days_since_base_out);
    end    
    clearvars days_since_base_out dataOut data lim_baseDate_string
    years(t)
    toc
end



end