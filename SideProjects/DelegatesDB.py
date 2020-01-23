import pandas as pd
import gender_guesser.detector as gender
'''
file = open('delagates.txt')
delagates = list(tuple())
current_group = ""
for i in file:
    if(i[0:3] == 'WIN'):
        current_group = i
        continue
    if(i[0] == '-'):
        fullname = i[2:].strip().split()
        if(len(fullname) > 3):
            if(fullname[0] == 'Bro' or fullname[0] == 'Sis'):
                delagates.append((fullname[2],fullname[1],current_group.strip(),('male' if fullname[len(fullname)-1] == 'm' else 'female')))
            elif("\"" in fullname[2]):
                delagates.append((fullname[3], fullname[0], current_group.strip(),('male' if fullname[len(fullname)-1] == 'm' else 'female')))
            else:
                delagates.append((fullname[1] + ' ' + fullname[2], fullname[0], current_group.strip(),('male' if fullname[len(fullname)-1] == 'm' else 'female')))
            continue
        delagates.append((fullname[1],fullname[0],current_group.strip(),('male' if fullname[len(fullname)-1] == 'm' else 'female')))

file.close()
total = len(delagates)
print(delagates)
print('total delegates: ',total)

file = open('reviseddelegates.txt')
for i in file:
    if(i[0:3] == 'WIN' or i.strip() == ""):
        continue
    print(i.strip().split(' – '))

database = pd.DataFrame(delagates,columns=('Last Name',"First Name",'Region','Gender'))

database['Arrival'] = ""
database['Departure'] = ""
database['WIN Churches Rep'] = 0
database['Housing Location'] = ""
database['Registration Fee Payment'] = ""
database['Pickup Person'] = ""


#database.to_excel('delegates.xlsx')
#database.to_csv('delegates.csv')
print(database)
regiongroup = database.groupby('Region')

for i in regiongroup:
    i = i[1].sort_values('Last Name')

'''
df = pd.read_excel('delegates.xlsx')
print(df)