classdef Meye

    properties (Access=private)
        model
    end


    methods

        % CONSTRUCTOR
        %------------------------------------------------------------------
        function self = Meye(modelPath)
            % Class constructor
            arguments
                modelPath char {mustBeText}
            end

            % Change the current directory to the directory where the
            % original class is, so that the package with the custom layers
            % is created there
            classPath = getClassPath(self);
            oldFolder = cd(classPath);
            % Import the model saved as ONNX
            self.model = importONNXNetwork(modelPath, ...
                'GenerateCustomLayers',true, ...
                'PackageName','customLayers_meye',...
                'InputDataFormats', 'BSSC',...
                'OutputDataFormats',{'BSSC','BC'});

            % Manually change the "nearest" option to "linear" inside of
            % the automatically generated custom layers. This is necessary
            % due to the fact that MATLAB still does not support the proper
            % translation between ONNX layers and DLtoolbox layers
            self.nearest2Linear([classPath filesep '+customLayers_meye'])

            % Go back to the old current folder
            cd(oldFolder)
        end


        % PREDICTION OF SINGLE IMAGES
        %------------------------------------------------------------------
        function [pupilMask, eyeProb, blinkProb] = predictImage(self, inputImage, options)
            % Predicts pupil location on a single image
            arguments
                self
                inputImage
                options.roiPos = []
                options.threshold = []
            end

            roiPos = options.roiPos;

            % Convert the image to grayscale if RGB
            if size(inputImage,3) > 1
                inputImage = im2gray(inputImage);
            end

            % Crop the frame to the desired ROI
            if ~isempty(roiPos)
                crop = inputImage(roiPos(2):roiPos(2)+roiPos(3)-1,...
                    roiPos(1):roiPos(1)+roiPos(4)-1);
            else
                crop = inputImage;
            end

            % Preprocessing
            img = double(imresize(crop,[128 128]));
            img = img / max(img,[],'all');

            % Do the prediction
            [rawMask, info] = predict(self.model, img);
            eyeProb = info(1);
            blinkProb = info(2);

            % Reinsert the cropped prediction in the frame
            if ~isempty(roiPos)
                pupilMask = zeros(size(inputImage));
                pupilMask(roiPos(2):roiPos(2)+roiPos(3)-1,...
                    roiPos(1):roiPos(1)+roiPos(4)-1) = imresize(rawMask, [roiPos(4), roiPos(3)],"bilinear");
            else
                pupilMask = imresize(rawMask,size(inputImage),"bilinear");
            end

            % Apply a threshold to the image if requested
            if ~isempty(options.threshold)
                pupilMask = pupilMask > options.threshold;
            end

        end


        % PREDICT A MOVIE AND GET A TABLE WITH THE RESULTS
        %------------------------------------------------------------------
        function tab = predictMovie(self, moviePath, options)
            % Predict an entire video file and returns a results Table
            %
            % tab = predictMovie(moviePath, name-value)
            % 
            % INPUT(S)
            %   - moviePath: (char/string) Full path of a video file.
            %   - name-value pairs
            %       - roiPos: [x,y,width,height] 4-elements vector defining a
            %       rectangle containing the eye. Works best if width and
            %       height are similar. If empty, a prediction will be done on 
            %       a full frame(Default: []).
            %       - threshold: [0-1] The pupil prediction is binarized based
            %       on a threshold value to measure pupil size. (Default:0.4)
            %
            % OUTPUT(S)
            %   - tab: a MATLAB table containing data of the analyzed video

            arguments
                self
                moviePath char {mustBeText}
                options.roiPos double = []
                options.threshold = 0.4;
            end

            % Initialize a video reader
            v = VideoReader(moviePath);
            totFrames = v.NumFrames;

            % Initialize Variables
            frameN = zeros(totFrames,1,'double');
            frameTime = zeros(totFrames,1,'double');
            binaryMask = cell(totFrames,1);
            pupilArea = zeros(totFrames,1,'double');
            isEye = zeros(totFrames,1,'double');
            isBlink = zeros(totFrames,1,'double');

            tic
            for i = 1:totFrames
                % Progress report
                if toc>10
                    fprintf('%.1f%% - Processing frame (%u/%u)\n', (i/totFrames)*100 , i, totFrames)
                    tic
                end

                % Read  a frame and make its prediction
                frame = read(v, i, 'native');
                [pupilMask, eyeProb, blinkProb] = self.predictImage(frame, roiPos=options.roiPos,...
                    threshold=options.threshold);

                % Save results for this frame
                frameN(i) = i;
                frameTime(i) = v.CurrentTime;
                binaryMask{i} = pupilMask > options.threshold;
                pupilArea(i) = sum(binaryMask{i},"all");
                isEye(i) = eyeProb;
                isBlink(i) = blinkProb;
            end
            % Save all the results in a final table
            tab = table(frameN,frameTime,binaryMask,pupilArea,isEye,isBlink);
        end



        % PREVIEW OF A PREDICTED MOVIE
        %------------------------------------------------------------------
        function predictMovie_Preview(self, moviePath, options)
            % Displays a live-preview of prediction for a video file

            arguments
                self
                moviePath char {mustBeText}
                options.roiPos double = []
                options.threshold double = []
            end
            roiPos = options.roiPos;


            % Initialize a video reader
            v = VideoReader(moviePath);
            % Initialize images to show
            blankImg = zeros(v.Height, v.Width, 'uint8');
            cyanColor = cat(3, blankImg, blankImg+255, blankImg+255);
            pupilTransparency = blankImg;

            % Create a figure for the preview
            figHandle = figure(...
                'Name','MEYE video preview',...
                'NumberTitle','off',...
                'ToolBar','none',...
                'MenuBar','none', ...
                'Color',[.1, .1, .1]);

            ax = axes('Parent',figHandle,...
                'Units','normalized',...
                'Position',[0 0 1 .94]);

            imHandle = imshow(blankImg,'Parent',ax);
            hold on
            cyanHandle = imshow(cyanColor,'Parent',ax);
            cyanHandle.AlphaData = pupilTransparency;
            rect = rectangle('LineWidth',1.5, 'LineStyle','-.','EdgeColor',[1,0,0],...
                'Parent',ax,'Position',[0,0,0,0]);
            hold off
            title(ax,'MEYE Video Preview', 'Color',[1,1,1])

            % Movie-Showing loop
            while exist("figHandle","var") && ishandle(figHandle) && hasFrame(v)
                try
                    tic
                    frame = readFrame(v);

                    % Actually do the prediction
                    [pupilMask, eyeProb, blinkProb] = self.predictImage(frame, roiPos=roiPos,...
                        threshold=options.threshold);

                    % Update graphic elements
                    imHandle.CData = frame;
                    cyanHandle.AlphaData = imresize(pupilMask, [v.Height, v.Width]);
                    if ~isempty(roiPos)
                        rect.Position = roiPos;
                    end
                    titStr = sprintf('Eye: %.2f%% - Blink:%.2f%% - FPS:%.1f',...
                        eyeProb*100, blinkProb*100, 1/toc);
                    ax.Title.String = titStr;
                    drawnow
                catch ME
                    warning(ME.message)
                    close(figHandle)
                end
            end
            disp('Stop preview.')
        end


    end

    
    %------------------------------------------------------------------
    %------------------------------------------------------------------
    % INTERNAL FUNCTIONS
    %------------------------------------------------------------------
    %------------------------------------------------------------------
    methods(Access=private)
        %------------------------------------------------------------------
        function path = getClassPath(~)
            % Returns the full path of where the class file is

            fullPath = mfilename('fullpath');
            [path,~,~] = fileparts(fullPath);
        end

        %------------------------------------------------------------------
        function [fplist,fnlist] = listfiles(~, folderpath, token)
            listing = dir(folderpath);
            index = 0;
            fplist = {};
            fnlist = {};
            for i = 1:size(listing,1)
                s = listing(i).name;
                if contains(s,token)
                    index = index+1;
                    fplist{index} = [folderpath filesep s];
                    fnlist{index} = s;
                end
            end
        end

        % nearest2Linear
        %------------------------------------------------------------------
        function nearest2Linear(self, inputPath)
            fP = self.listfiles(inputPath, 'Shape_To_Upsample');

            foundFileToChange = false;
            beforePatter = '"half_pixel", "nearest",';
            afterPattern = '"half_pixel", "linear",';
            for i = 1:length(fP)

                % Get the content of the file
                fID = fopen(fP{i}, 'r');
                f = fread(fID,'*char')';
                fclose(fID);

                % Send a verbose warning the first time we are manually
                % correcting the upsampling layers bug
                if ~foundFileToChange && contains(f,beforePatter)
                    foundFileToChange = true;
                    msg = ['This is a message from MEYE developers.\n' ...
                        'In the current release of the Deep Learning Toolbox ' ...
                        'MATLAB does not translate well all the layers in the ' ...
                        'ONNX network to native MATLAB layers. In particular the ' ...
                        'automatically generated custom layers that have to do ' ...
                        'with UPSAMPLING are generated with the ''nearest'' instead of ' ...
                        'the ''linear'' mode.\nWe automatically correct for this bug when you ' ...
                        'instantiate a Meye object (henche this warning).\nEverything should work fine, ' ...
                        'and we hope that in future MATLAB releases this hack wont be ' ...
                        'needed anymore.\n' ...
                        'If you find bugs or performance issues, please let us know ' ...
                        'with an issue ' ...
                        '<a href="matlab: web(''https://github.com/fabiocarrara/meye/issues'')">HERE.</a>'];
                    warning(sprintf(msg))
                end

                % Replace the 'nearest' option with 'linear'
                newF = strrep(f, beforePatter, afterPattern);

                % Save the file back in its original location
                fID = fopen(fP{i}, 'w');
                fprintf(fID,'%s',newF);
                fclose(fID);
            end
        end
    end
end


