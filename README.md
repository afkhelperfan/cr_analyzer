# cr_analyzer
auto detecting cr results with OCR.

## dependencies
### Python  
pytesseract
opencv-python
numpy
matplotlib
pandas
labelImg  

### System
tesseract  
sqlite3  

## setup
1. Install dependencies.
2. Use 4K screenshots as possible. If you are using emulator, this can be tweaked easily in settings.
3. put the screenshots of the statistics as these format
data/{trial}/{round}.png

2. do labeling of the results statistics and save as dmg_data_4k.json.
The labels are dps_1, dps_2, dps_3, dps_4, dps_5 from 1-5 position.
It only works on fixed resolution setup, so this needs to be setup again if you use another smartphone or friend's sceenshot.  

3. Enter what character did you use in the char_label/{round}.json. This feature would be refined by editing in gui in future.  

4. Modify the db char_data.db using sqlite browser whether the character is carry or not. This feature would be refined by editing in gui in future.  
carry value 3 : 1B+ stable carry  
carry value 2 : sub carry or low dps main carry
carry value 1 : sub carry  
carry value 0 : non carry

5. execute this command
```python  
python3 cr_analyzer.py [trial]
```

6. To get the reports, execute this command
```python  
python3 trial_summary.py
```  

