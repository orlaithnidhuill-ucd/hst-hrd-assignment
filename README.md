This repo is for my Optical Detectors assignment where the aim was to build a Hertzsprung–Russell Diagram (H–R Diagram) using HST data (filters F336W and F555W).

    ...........................................................................................
    
The main steps in the code are:
-Combining the aligned FITS images (median stack)
-Finding stars using a simple local-max and sigma-clipped stats method
-Doing aperture photometry
-Converting flux to magnitudes
-Plotting the final Hertzprung Russell diagram

Most of the code, logic and parameters come straight from the Brightspace worked workbooks 1–3.


    ...........................................................................................


FOLDERS...
¦-data/
¦---F336W/
¦---F555W/
¦-ccd_utils.py
¦-your-code.py
¦-README.md

When you run the script it also creates an “outputs” folder where all results (combined FITS, catalog, plots) are saved.

    .........................................................................................
    
INSTRUCTIONS....
Clone the repo and just run the main script;

    git clone [https://github.com/orlaithnidhuill-ucd/hst-hrd-assignment.git]
    
    cd hst-hrd-assignment
    
    python your-code.py

See the screenshot below;

<img width="567" height="196" alt="image" src="https://github.com/user-attachments/assets/869c8b6b-e099-49be-8b49-1ae5b3562420" />

The results should be output like this; 

<img width="578" height="463" alt="image" src="https://github.com/user-attachments/assets/c7d5d0c6-fe21-4aa1-a456-0ef1ecc9d709" />

<img width="622" height="491" alt="image" src="https://github.com/user-attachments/assets/c5f2a3d2-9221-4b42-b63a-a92330d72799" />

    ................................................................................................

Assistance statement;
In alignment with academic integrity, I declare that I used AI (ChatGPT) and StackOverflow a few times for debugging help and for improving outputs and portability.
All the actual logic, parameters, and workflow follow the uploaded course Brightspace notebooks, so the work is my own with guidance from Worked Workbooks 1-3. Plagarism was avoided where possible.
