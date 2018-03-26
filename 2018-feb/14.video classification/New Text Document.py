def preprocess_videos(base_dir, target_dir):
    if not os.path.exists(base_dir): 
        print(base_dir, ' does not exist')
        return
    if os.path.exists(target_dir): 
        print(target_dir, ' exists and continuing with existing data')
        return   
    os.makedirs(target_dir)      
    categories = os.listdir(base_dir)
    for category in categories:
        video_category_path = os.path.join(base_dir, category)    
        video_listings = os.listdir(video_category_path)
        frames_category_path = os.path.join(target_dir, category)
        count = 1
        for file in video_listings[0:2]:
            video = cv2.VideoCapture(os.path.join(video_category_path,file))
            #print(video.isOpened())
            framerate = video.get(5)
            frames_path = os.path.join(frames_category_path, "video_" + str(int(count)))
            os.makedirs(frames_path)
            while (video.isOpened()):
                frameId = video.get(1)
                success,image = video.read()
                if(success == False):
                    break
                image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
                if (frameId % math.floor(framerate) == 0):
                    filename = os.path.join(frames_path, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                    print(filename)
                    cv2.imwrite(filename,image)
            video.release()
            count+=1

preprocess_videos('C:\\Users\\Thimma Reddy\\videos-keras', 
                  'C:\\Users\\Thimma Reddy\\data2')
