# cr_analyzer
auto detecting AFK Arena CR Results with OCR.

## Dependencies
### Python  
pytesseract  
opencv-python  
numpy  
matplotlib  
pandas  
labelImg  
pytorch  
pytorch_msssim    

### System
tesseract    
sqlite3   

## Usage  
### Results Scanning  
```bash
 python3 cr_analyzer.py --bin_thres 200 [boss] [username] [start] [end] [lang]
 ```
 boss param : boss name. a folder named data/[boss] needs to exist.  
 username param : username. a folder named data/[boss]/[username] needs to exist.  
 start param : trial to start scanning. It need to be in natural number. A folder named data/[boss]/[username]/[start] needs to exist.  
 end param : trial to end scanning. It need to be in natural number. A folder named data/[boss]/[username]/[start]  ~ data/[boss]/[username]/[end] needs to exist.  

### Line Detection Option  
This feature enables scanning without providing any screenshot annotation data. It only works on JP/EN/CN AFK Arena font. 
```bash
 python3 cr_analyzer.py --line_detect --bin_thres 200 [boss] [username] [start] [end] [lang]
 ``` 

### Comp Detection Option  
This feature enables automatic comp scanning. This feature makes scanning very slow, and does not support skinned characters.   
```bash 
 python3 cr_analyzer.py --line_detect --comp_detect --full_scan --bin_thres 200 [boss] [username] [start] [end] [lang]
```  
If you are trying to scan inhomogenious screenshot results(different resolution, different comps, different user), enable the following option.  
```bash 
 python3 cr_analyzer.py --line_detect --comp_detect --full_scan --bin_thres 200 [boss] [username] [start] [end] [lang]
```  

### Tree Detection  
TBD    

#### Requirements  
1. Each trial needs a tree information called tree.json.   
2. Each boss needs a comp information. It is in the data/[boss]/[username]/char_label folder, and the comp information is provided as 1.json ~ 6.json. This information could be omitted by enabling comp detection option in the cr_analyzer.py. 
3. Each trial screenshot needs to be renamed to 1.png/jpg to 6.png/jpg to the corresponding comp id. This can be done by rename_img_files.ps1 and rename_img_files.bash depending on the OS. 
4. Damage Results Screenshot location should be annotated. This information is provided in ata/[boss]/[username]/dmg_data.json. This information could be omitted by enabling line detection option in the cr_analyzer.py.  

  
### Results Scanning with Line Detection
```bash
 python3 cr_analyzer.py --line_detect --bin_thres 200 [boss] [username] [start] [end] [lang]
 ```


## Setup (Recommended)
1. Install dependencies.
2. Use 4K screenshots as possible. If you are using emulator, this can be tweaked easily in settings.
3. put the screenshots of the statistics as these format
data/{boss name}/{username}/{trial}/{round}.png

4. do labeling of the results statistics and save as dmg_data.json.
The labels are dps_1, dps_2, dps_3, dps_4, dps_5 from 1-5 position.
It only works on fixed resolution setup, so this needs to be setup again if you use another smartphone or friend's sceenshot.  

```bash
labelImg
```
![labelimg](labelimg.png)


3. Enter what character did you use in the data/char_label/{round}.json. This feature would be refined by editing in gui in future.  
```json
{"dps_1" : "raku", "dps_2": "grezhul", "dps_3" : "rosaline", "dps_4" : "estrilda", "dps_5" : "twins"}
```

4. Modify the db char_data.db using sqlite browser and add new character and roles.  
![db](db.png)

## Run

1. convert the screenshot name to [1-6].png
```bash
./rename_img_files.bash [boss] [trial_start] [trial_end]
```  

2. add tree information as tree.json in each trial.  
```json
{"sus" : 130, "fort" : 108, "cele" : 150, "might": 5, "sorc" : 152}
```

3. execute this command  
```bash
./analyze_cr_batch.bash [boss] [trial_start] [trial_end]
```

4. To get the reports, execute this command
```python  
python3 trial_summary.py
```  
The reports are generated in reports/ folder.

![corr](reports/sorc-corr.png)

![char](reports/sus-0.0fort-0.0sorc-0.0cele-0.0might-0.0--character.png)

![overview](reports/sus-0.0fort-0.0sorc-0.0cele-0.0might-0.0--overview.png)

## WIP
```python
python3 image_matcher.py
```
able to recognize comp character information.  

## Roadmap  
- [ ] Role - Sum based correlation
- [ ] Calculate RNG and make a list of damage range in probability
- [ ] Adding new metrics : Tree - Comp Nonlinear relationship
- [ ] Adding new metrics : Power Ratings
- [x] Making High Precision Character Matching
- [x] Finding Searching Strategy for Tree stats

