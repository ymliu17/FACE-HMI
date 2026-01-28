import os
import pandas as pd
from datetime import datetime
import json

def parse_datetime(dt_str):
    try:
        parsed_date = datetime.strptime(dt_str, '%m/%d/%Y %H:%M:%S:%f')
        return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            parsed_date = datetime.strptime(dt_str, '%m/%d/%Y %I:%M:%S %p')
            return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                parsed_date = datetime.strptime(dt_str, '%m/ %d/%Y  %H:%M:%S:%f')
                return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                return None

def build_RT(rt_csv):
    df_rt = pd.read_csv(rt_csv)
    columns = ['Sub_Ses_Block', 'Subject', 'Session', 'Game', 'Level', 'Accuracy', 'Block', 'StartT', 'EndT', 'Mean_RT', 'Std_RT']
    rt_labels = pd.DataFrame(columns=columns)

    for (subject, session), group in df_rt.groupby(['study_id', 'session']):
        block_index = 1
        sep_index = 0
        miss_time = 0
        for _, row in group.iterrows():
            game = row['game']
            level = row['start_level']
            if level > 1000:
                level = -1
            else:
                level = level // 10 % 10
            accuracy = row['accuracy']
            results = json.loads(row['results'])['results']
      
            block_start = parse_datetime(results[0]['trialStartTime'])
            block_end = parse_datetime(results[-1]['trialEndTime'])
            
            vid_start = parse_datetime(group.iloc[0]['created'])
            
            if row['id'] == group.iloc[sep_index]['id']:
                video_end = parse_datetime(row['completed'])
            else:
                sep_vid_start = parse_datetime(row['created'])
                miss_time += (pd.to_datetime(sep_vid_start) - pd.to_datetime(video_end)).total_seconds()
                sep_index = block_index - 1
            
            startT = pd.to_datetime(block_start) + pd.Timedelta(hours=7) - pd.to_datetime(vid_start)
            endT = pd.to_datetime(block_end) + pd.Timedelta(hours=7) - pd.to_datetime(vid_start)
                
            if startT.total_seconds() < 0:
                startT = startT + pd.Timedelta(hours=1)
                endT = endT + pd.Timedelta(hours=1)
            startT = startT.total_seconds() - miss_time
            endT = endT.total_seconds() - miss_time

            if game == 'Mot':
                df_trial = pd.DataFrame(results)
                df_id = df_trial.groupby('id')
                filtered_ids= df_id.filter(lambda x: (x['res'] == 'S').all())['id'].unique()
                filtered_rt = []
                for trial_id in filtered_ids:
                    last_reaction_time = df_trial[df_trial['id'] == trial_id]['reactionTime'].iloc[-1]
                    filtered_rt.append({'id': trial_id, 'reactionTime': last_reaction_time})

            elif game == 'Sound Sweeper':
                df_trial = pd.DataFrame(results)
                df_id = df_trial.groupby('id')
                filtered_ids= df_id.filter(lambda x: (x['res'] == 'S').all())['id'].unique()
                filtered_rt = []
                for trial_id in filtered_ids:
                    total_reaction_time = df_trial[df_trial['id'] == trial_id]['reactionTime'].sum()
                    filtered_rt.append({'id': trial_id, 'reactionTime': total_reaction_time})

            else:
                filtered_rt = [trial for trial in results if trial['res'] == 'S']
            
            reaction_times = [trial_rt['reactionTime'] / 1000 for trial_rt in filtered_rt if trial_rt['reactionTime'] > 0]
            mean_RT = sum(reaction_times) / len(reaction_times) if reaction_times else 0
            std_RT = pd.Series(reaction_times).std() if reaction_times else 0

            rt_label = [
                f"{subject}_{session}_{block_index}",
                subject,
                session,
                game,
                level,
                accuracy,
                block_index,
                startT,
                endT,
                mean_RT,
                std_RT
            ]
            rt_labels.loc[len(rt_labels)] = rt_label
            block_index += 1

    return rt_labels


def build_fatigue(fatigue_csv):
    df = pd.read_csv(fatigue_csv)
    columns = ['Sub_Ses_Block', 'Concept', 'Focus', 'Difficulty', 'Novelty', 'Tiredness', 'PreFatigue', 'PostFatigue']
    fatigue_labels = pd.DataFrame(columns=columns)

    for index, row in df.iterrows():
        subject = row['study_id']
        session = row['session']
        concept = 'Staircase' if row['concept_type'] == 1 else 'Novelty'

        pre_fatigue_columns = ['fatigue_pre_1', 'fatigue_pre_2', 'fatigue_pre_3', 'fatigue_pre_4', 'fatigue_pre_5', 'fatigue_pre_6']
        pre_fatigue = float(row[pre_fatigue_columns].mean())
        post_fatigue_columns = ['fatigue_post_1', 'fatigue_post_2', 'fatigue_post_3', 'fatigue_post_4', 'fatigue_post_5', 'fatigue_post_6']
        post_fatigue = float(row[post_fatigue_columns].mean())
        
        for block_idx in range(1, 26):
            focus = row[f'focus_block_{block_idx}']
            difficulty = row[f'difficult_b{block_idx}']
            novolty = row[f'novel_block_{block_idx}']
            tiredness = row[f'tired_block_{block_idx}']
        
            fatigue_label = [subject+'_'+str(session)+'_'+str(block_idx), concept,focus, difficulty, novolty, tiredness, pre_fatigue, post_fatigue]
            fatigue_labels.loc[len(fatigue_labels)] = fatigue_label
    
    fatigue_labels.replace(99, -1, inplace=True)
    fatigue_labels.dropna(inplace=True)
  
    return fatigue_labels


def main():
    rt_csv = '/Users/yangliu/Programs/FACE/FACE_cognitive_block_trial_data.csv'
    rt_labels = build_RT(rt_csv)
    # rt_labels.to_csv('rt_label.csv', index=False)
    
    fatigue_csv = '/Users/yangliu/Programs/FACE/FACE_intervention_questionnaire_data.csv'
    fatigue_labels = build_fatigue(fatigue_csv)
    # fatigue_labels.to_csv('fatigue_label.csv', index=False)

    vid_labels = rt_labels.merge(fatigue_labels, on='Sub_Ses_Block', how='inner')
    cols = ['Sub_Ses_Block', 'Subject', 'Session', 'Block', 'Game', 'Level', 'Accuracy','Concept', 'StartT', 'EndT', 'Mean_RT', 'Std_RT', 'Focus', 'Difficulty', 'Novelty', 'Tiredness', 'PreFatigue', 'PostFatigue']
    vid_labels = vid_labels[cols]

    vid_labels = vid_labels[vid_labels['Std_RT'] != 'nan']
    vid_labels = vid_labels[vid_labels['Tiredness'] != -1]
    vid_labels = vid_labels[vid_labels['PostFatigue'] != -1]
    vid_labels = vid_labels[vid_labels['Difficulty'] != -1]
    vid_labels = vid_labels[vid_labels['StartT'] > 0]
    
    vid_labels.to_csv('vid_labels.csv', index=False)
    print(f'length of video_labels: ', len(vid_labels))



if __name__ == "__main__":
    main()