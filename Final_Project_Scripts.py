'''
Alexis Parra
SL: Stephen Kim
04/26/21
ISTA 350 Final Project - Pokemon with stats
Overview: Webscrapes https://pokemondb.net/pokedex/all for pokemon data
Group Members: AJ Gregg, Winston Cox, Kevin Ho
''' 
import pandas as pd
import requests, statsmodels.api as sm
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.pyplot as plt, numpy as np
from numpy.polynomial.polynomial import polyfit
from bs4 import BeautifulSoup

class HTMlTableParser:
    '''
    Parses the HTML table from the pokemon DB site into a pandas DataFrame.
    Code provided by AJ.
    '''
    def parse_url(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'lxml')
        #print("\nPage Status (200 is successful): " + str(page.status_code))

        return [(table['id'],self.parse_html_table(table))\
            for table in soup.find_all('table')]
        
    def parse_html_table(self, table):
        '''
        Parses the HTML table
        '''
        n_columns = 0
        n_rows=0
        column_names = []
    
            # Find number of rows and columns
            # we also find the column titles if we can
        for row in table.find_all('tr'):
                # Determine the number of rows in the table
            td_tags = row.find_all('td')
            if len(td_tags) > 0:
                n_rows+=1
                if n_columns == 0:
                        # Set the number of columns for our table
                    n_columns = len(td_tags)
                        
                # Handle column names if we find them
            th_tags = row.find_all('th') 
            if len(th_tags) > 0 and len(column_names) == 0:
                for th in th_tags:
                    column_names.append(th.get_text())
    
            # Safeguard on Column Titles
        if len(column_names) > 0 and len(column_names) != n_columns:
            raise Exception("Column titles do not match the number of columns")
    
        columns = column_names if len(column_names) > 0 else range(0,n_columns)
        df = pd.DataFrame(columns = columns,
                              index= range(0,n_rows))
        row_marker = 0
        for row in table.find_all('tr'):
            column_marker = 0
            columns = row.find_all('td')
            for column in columns:
                df.iat[row_marker,column_marker] = column.get_text()
                column_marker += 1
            if len(columns) > 0:
                row_marker += 1
                    
            # Convert to float if possible
        for col in df:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass
            
        return df

def scatter_plot(df):
    '''
    Are pokemons roughly created to have a balanced health and defense?
    Gets Pearon's r-value to asses hypothesis 
    '''
    atk_df = df['Attack']
    def_df = df['Defense']
    hitpoints_df = df['HP']
    x = hitpoints_df
    y = atk_df
    z = def_df
    plt.plot(x,y,'o',markersize=2,color='r')
    plt.plot(x,z,'o',markersize=2,color= 'green')
    m,b = np.polyfit(x,y, 1)
    q,r = np.polyfit(x,z, 1) 
    plt.plot(x, m*x + b, markersize=5,color='lightcoral') # graphs linear regressions
    plt.plot(x, q*x + r, markersize=5,color='lime')
    plt.title('All Pokemons Attack & Defense Relationship', fontsize=15)
    plt.xlabel('Hitpoints', fontsize=20)
    plt.ylabel('Attack & Defense Stat', fontsize=20)
    red_patch = mpatches.Patch(color='red', label='Attack')
    green_patch = mpatches.Patch(color='green', label='Defense')
    plt.legend(handles=[red_patch, green_patch], loc='upper left') 

    corr_matrix = np.corrcoef(x, y) # calculates r vals for atk and def
    corr_xy = corr_matrix[0,1]
    atk_rval = round(corr_xy**2,5)

    corr_matrix = np.corrcoef(x, z)
    corr_xz = corr_matrix[0,1]
    def_rval = round(corr_xz**2,5)
    
    plt.text(140,240,'R-values:',fontsize=20, color='black')
    plt.text(175,200,atk_rval,fontsize=20, color='red')
    plt.text(175,225,def_rval,fontsize=20, color='green')


def bar_graph(df):
    '''
    Bar graph comparison of pokemon types Ground vs Flying
    '''
    #Ground Type DF
    ground_df = df[df['Type'].str.contains("Ground")]   #creates df with all ground types
    ground_stat_vals = []
    stat_names = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    index = ground_df.index #Gets the indexs of the df 
    num_of_ground_pokemon = len(index) #Gets the number of pokemon in df

    ground_avg_hp_stat = (ground_df['HP'].sum())/num_of_ground_pokemon    # Combines all the stat, then divides by amount of pokemon in df
    ground_avg_atk_stat = (ground_df['Attack'].sum())/num_of_ground_pokemon
    ground_avg_def_stat = (ground_df['Defense'].sum())/num_of_ground_pokemon
    ground_avg_spatk_stat = (ground_df['Sp. Atk'].sum())/num_of_ground_pokemon
    ground_avg_spdef_stat = (ground_df['Sp. Def'].sum())/num_of_ground_pokemon
    ground_avg_spd_stat = (ground_df['Speed'].sum())/num_of_ground_pokemon
    ground_avg_total_stat = (ground_df['Total'].sum())/num_of_ground_pokemon
    
    ground_stat_vals.append(round(ground_avg_hp_stat, 2)) # Appends stats into stat_values list in specific order
    ground_stat_vals.append(round(ground_avg_atk_stat,2))
    ground_stat_vals.append(round(ground_avg_def_stat,2))
    ground_stat_vals.append(round(ground_avg_spatk_stat,2))
    ground_stat_vals.append(round(ground_avg_spdef_stat,2))
    ground_stat_vals.append(round(ground_avg_spd_stat,2))
    ground_avg_total_stat = (round(ground_avg_total_stat,2))

    #Flying DF
    flying_df = df[df['Type'].str.contains("Flying")]   #creates df with all ground types
    flying_stat_vals = []
    #stat_names = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    index = flying_df.index #Gets the indexs of the df 
    num_of_flying_pokemon = len(index) #Gets the number of pokemon in df

    flying_avg_hp_stat = (flying_df['HP'].sum())/num_of_flying_pokemon    # Combines all the stat, then divides by amount of pokemon in df
    flying_avg_atk_stat = (flying_df['Attack'].sum())/num_of_flying_pokemon
    flying_avg_def_stat = (flying_df['Defense'].sum())/num_of_flying_pokemon
    flying_avg_spatk_stat = (flying_df['Sp. Atk'].sum())/num_of_flying_pokemon
    flying_avg_spdef_stat = (flying_df['Sp. Def'].sum())/num_of_flying_pokemon
    flying_avg_spd_stat = (flying_df['Speed'].sum())/num_of_flying_pokemon
    flying_avg_total_stat = (flying_df['Total'].sum())/num_of_flying_pokemon
    
    flying_stat_vals.append(round(flying_avg_hp_stat, 2)) # Appends stats into stat_values list in specific order
    flying_stat_vals.append(round(flying_avg_atk_stat,2))
    flying_stat_vals.append(round(flying_avg_def_stat,2))
    flying_stat_vals.append(round(flying_avg_spatk_stat,2))
    flying_stat_vals.append(round(flying_avg_spdef_stat,2))
    flying_stat_vals.append(round(flying_avg_spd_stat,2))
    flying_avg_total_stat = (round(flying_avg_total_stat,2))

    #Bar Plot Creation
    barWidth= 0.25
    r1 = np.arange(len(ground_stat_vals))
    r2 = [x + barWidth for x in r1] #Creates off set for 2nd column

    x = np.arange(len(stat_names))
    fig, ax = plt.subplots()
    rects1 = plt.bar(x - barWidth/2, ground_stat_vals, barWidth, label='Ground', color= 'sandybrown')  #Creates Ground Type Bars
    rects2 = plt.bar(x + barWidth/2, flying_stat_vals, barWidth, label='Flying', color='skyblue')    #Creates Flying Type Bars

    ax.set_ylabel('Average Stat Total', labelpad=50, fontsize=15, rotation=90) #Adds y-axis label
    ax.set_title("Pokemon (Ground Type VS Flying Type)\n Pokemon Average Stats:\n Num of Ground Pokemon: "+str(num_of_ground_pokemon)+
                "\nAverage Ground Total Stat: "+str(round(ground_avg_total_stat))+"\nNum of Flying Pokemon: "+str(num_of_flying_pokemon)+
                "\nAverage Flying Total Stat: "+str(round(flying_avg_total_stat)), loc="center", fontsize=10)   #Adds title
    ax.set_xticks(x)
    ax.set_xticklabels(stat_names)
    ax.set_xlabel("Pokemon Statistics", fontsize=15, rotation=0) #Adds x-axis label
    ax.legend(loc="upper left")
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

def radar_chart(df):
    '''
    Radar Chart of popular pokemons
    '''
    labels = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    legends = []
    for i in range(1,10, 2):
        data = df.loc[i, labels].values
        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        stats=np.concatenate((data,[data[0]]))
        angles=np.concatenate((angles,[angles[0]]))

        xs = range(6)
        ax.set_xticks(xs)
        ax.set_xticklabels(["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])
        ax.plot(angles, stats, "o-")
        ax.fill(angles, stats, alpha=0.25)
        #ax.set_thetagrids(np.degrees(angles), labels)
        legends.append(df.loc[i, "Name"])
        
    plt.title("Radar Chart\n of\nPokemon Stats") 
    plt.legend(legends, loc=('lower right')) 


# Main method
def main():
    url = "https://pokemondb.net/pokedex/all"
    hp = HTMlTableParser()
    table = hp.parse_url(url)[0][1] #creates a table and parses it into a DF
    #table.to_excel("Pokemon Data Table.xlsx")

    scatter_plot(table)
    bar_graph(table)
    radar_chart(table)
    plt.show()

if __name__ == '__main__':
    main()

