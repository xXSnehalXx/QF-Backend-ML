1. Run the following commands at once 

chmod +x server.sh
chmod +x packages.sh
chmod +x install.sh


2. Install Server and Requirements, Run the following command

./install.sh

3. Run the server 

./server.sh

4. Perform endpoint requests

    1. Train Model (GET)

        http://0.0.0.0:8000/csrs/train
        http://0.0.0.0:8000/csrs/loadModel

    2. Predict Cross Selling (POST)

        http://0.0.0.0:8000/csrs/cs

        append the following in body of request 


        [{"QF_MERCHANT_PRID": 1507, "QF_MERCHANT_SERVICE_PRID": [1311, 1311, 1247], "SERVICE_NAME": ["Blowdry Outcurls / Flipouts", "Blowdry Outcurls / Flipouts", "Class 6 Tuition"]},
                {"QF_MERCHANT_PRID": 1892, "QF_MERCHANT_SERVICE_PRID": [2162, 2118, 2119], "SERVICE_NAME": ["Full kitchen + Appliances (Without ", "Deep Bathroom cleaning", "2 BHK Furnished Apartment deep clea"]}, 
                {"QF_MERCHANT_PRID": 1979, "QF_MERCHANT_SERVICE_PRID": [2368, 2368, 2368], "SERVICE_NAME": ["Bike - Foam Wash", "Bike - Foam Wash", "Bike - Foam Wash"]}, 
                {"QF_MERCHANT_PRID": 1948, "QF_MERCHANT_SERVICE_PRID": [2297, 2295, 2299], "SERVICE_NAME": ["Ceiling Light Installation", "Split AC Uninstallation", "Fan Repair"]}]      


    (No need to always load model for predicting, whenever server starts the model already loads, but only required once in the starting)

5. Whenever new requirements are added, run the following command in the terminal

./packages.sh