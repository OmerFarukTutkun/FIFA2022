
import torch
groups = [
   ['Qatar' , 'Ecuador' ,'Senegal' , 'Netherlands'],
   ['England' , 'Iran' ,'United States' , 'Wales'],
   ['Argentina' , 'Saudi Arabia' ,'Mexico' , 'Poland'],
   ['France' , 'Australia' ,'Denmark' , 'Tunisia'],
   ['Spain' , 'Germany' ,'Japan' , 'Costa Rica'],
   ['Belgium' , 'Canada' ,'Morocco' , 'Croatia'],
   ['Brazil' , 'Serbia' ,'Switzerland' , 'Cameroon'],
   ['Portugal' , 'Ghana' ,'Uruguay' , 'South Korea'],
   ]
group_names="ABCDEFGH"
def FifaPredictLastN(model , train_dataset, device, teams):
    num_of_match = int(len(teams)/2)
    remaning = [0]* num_of_match
    for i in range(num_of_match):
        score = 0.0
        x = train_dataset.getTensorFromTeams(teams[2*i] , teams[2*i +1])
        with torch.no_grad():
            score = model(x.to(device))[0]

        x = train_dataset.getTensorFromTeams(teams[2*i +1 ] , teams[2*i])
        with torch.no_grad():
            score += 1- model(x.to(device))[0]
        if score > 1.0:
            winner = teams[2*i]
        else:
            winner = teams[2*i + 1] 
        print(teams[2*i]  + "  -  " + teams[2*i +1 ] + "  -> " + winner)
        remaning[i] = winner
    return remaning

def FifaPredict(model , train_dataset, device):
    model.eval()

    group_points = []
    last_16 = [0] * 16
    last_16_path = [ [0 , 9] , [8 , 1], [2 , 11], [10 , 3], [4 , 13], [12 , 5], [6 , 15], [14 , 7]]
    print("\n")
    for index, group in enumerate( groups):
        points = [ 0 , 0, 0, 0]
        for j in range(3):
            for k in range(j+1 ,4):
                result = 0.0
                x = train_dataset.getTensorFromTeams(group[j] , group[k])
                with torch.no_grad():
                    result = model(x.to(device))[0]

                x = train_dataset.getTensorFromTeams(group[k] , group[j])
                with torch.no_grad():
                    result =( 1- model(x.to(device))[0] +result)/2

                if result > 0.6:
                    points[j] += 3 + result/100
                elif result < 0.4:
                    points[k] += 3 + (1-result)/100
                else:
                    points[k] += 1 + (1 - result)/100
                    points[j] += 1 + result/100
        sorted_teams = sorted(range(len(points)), key=lambda k: points[k])
        print("Group " + group_names[index] +":")
        for i in range(1,5):
            idx = sorted_teams [-i]
            print(group[idx] +f"  {int(points[idx])}")
        print("\n")
       
        last_16[last_16_path[index][0]] = group[sorted_teams[-1]]
        last_16[last_16_path[index][1]] = group[sorted_teams[-2]]
    print("\nLast 16")
    last_8 = FifaPredictLastN(model , train_dataset, device, last_16)
    print("\n\nQuarter-finals")
    last_4 = FifaPredictLastN(model , train_dataset, device, last_8)
    print("\n\nSemi-finals")
    last_2 = FifaPredictLastN(model , train_dataset, device, last_4)
    print("\n\nFinal")
    winner = FifaPredictLastN(model , train_dataset, device, last_2)
    model.train()