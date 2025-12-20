import pandas as pd
import time
from pathlib import Path
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


seasons = {
    2025: "250044",
    2024: "240050",
    2023: "230054",
    2022: "220069",
    2021: "210064",
    2020: "200061",
    2019: "190075",
    2018: "180084",
    2017: "170099",
    2016: "160058",
    2015: "150059",
    2014: "140052", 
    2013: "130034",
    2012: "120002",
    2011: "110002",
    2010: "100002",
    2009: "090002",
    2008: "080002",
    2007: "070002"
}

# Use absolute path from this file's location
SAVE_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "raw_sofifa" / "yearly"


class SoFIFAScraper:
    def __init__(self, seasons, base_url, save_dir):
        self.seasons = seasons
        self.base_url = base_url
        self.save_dir = save_dir

    def initialize_driver(self):        
        chrome_options = Options()

        chrome_options.add_experimental_option("detach", True)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])

        # Add user agent to avoid bot detection
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        # Performance optimizations
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-extensions')
        
        chrome_options.page_load_strategy = "eager"

        driver = webdriver.Chrome(options=chrome_options)
        print("Chrome WebDriver initialized")
        
        return driver


    def get_team_stats(self, driver, season_code, year):
        """
        Scrape player stats for a given season.
        
        Args:
            driver: Selenium WebDriver instance
            season_code: SoFIFA season code (e.g., "240050")
            year: Calendar year (e.g., 2024)
            
        Returns:
            DataFrame with all players for the season
        """
        data_list = []  # Use list instead of DataFrame for efficiency
        offset = 0
        
        while True:
            try:
                driver.get(f"{self.base_url}&r={season_code}&offset={offset}")
                time.sleep(3)  # Wait for page load
                
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                
            except TimeoutException:
                print(f"No more data at offset {offset}. Scraping complete!")
                break
            except Exception as e:
                print(f"Error at offset {offset}: {e}")
                break
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            table = soup.find('table')
            
            if not table:
                print("ERROR: No table found!")
                break
            
            # Get headers only once
            if offset == 0:
                headers = [th.get_text(strip=True) for th in table.select("thead th")]
            
            # Extract rows
            for row in table.select("tbody tr"):
                cols = row.find_all(['th', 'td'])
                if cols:
                    data_list.append([col.get_text(strip=True) for col in cols])

            offset += 60
            
            if offset % 600 == 0:
                print(f"  {offset} players collected so far...")
        
        # Create DataFrame once at the end
        df = pd.DataFrame(data_list, columns=headers)
        
        # Add season information
        df['season'] = year
        df['season_code'] = season_code

        print(f"\n{'='*60}")
        print(f"Total rows collected: {len(df)}")
        print(f"{'='*60}")
        return df


    
    def close_driver(self, driver):
        """Close the WebDriver."""
        driver.quit()
        print("Chrome WebDriver closed")
    
    def collect_all(self):
        """Combine all yearly CSV files into one master file."""
        df_list = []
        
        for year in self.seasons.keys():
            file_path = self.save_dir / f"sofifa_players_{year}.csv"
            if file_path.exists():
                yearly_df = pd.read_csv(file_path)
                df_list.append(yearly_df)
            else:
                print(f"Warning: {file_path} not found, skipping...")
        
        # Combine all DataFrames at once
        df = pd.concat(df_list, ignore_index=True)
        
        output_path = self.save_dir.parent / "sofifa_players_all_years.csv"
        df.to_csv(output_path, index=False)
        
        print(f"All years combined ({len(df)} rows) and saved to: {output_path}")


    def yearly_scrape(self):
        """Scrape all seasons and save individual year files."""
        # Ensure directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        driver = self.initialize_driver()
        
        try:
            for year, season_code in self.seasons.items():
                print(f"\n{'='*60}")
                print(f"Starting year {year} (season code: {season_code})")
                print(f"{'='*60}")
                
                df = self.get_team_stats(driver, season_code, year)
                output_path = self.save_dir / f"sofifa_players_{year}.csv"
                df.to_csv(output_path, index=False)
                print(f"✓ {year} data saved to: {output_path}\n")
            
        
        finally:
            # Always close driver, even if error occurs
            self.close_driver(driver)
    
    def scrape_all_combined(self):
        """Scrape all seasons and write directly to one combined file."""
        # Ensure directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = self.save_dir.parent / "sofifa_players_all_years.csv"
        driver = self.initialize_driver()
        
        first_year = True
        total_rows = 0
        
        try:
            for year, season_code in self.seasons.items():
                print(f"\n{'='*60}")
                print(f"Starting year {year} (season code: {season_code})")
                print(f"{'='*60}")
                
                df = self.get_team_stats(driver, season_code, year)
                
                # Write to combined file (append after first year)
                
                df.to_csv(
                    output_path,
                    mode='w' if first_year else 'a',
                    header=first_year,
                    index=False
                )
                
                total_rows += len(df)
                print(f"✓ {year} data ({len(df)} rows) appended to combined file\n")
                
                # Clear memory
                del df
                first_year = False
        
        finally:
            self.close_driver(driver)
            
        print(f"\n{'='*60}")
        print(f"✓ All data combined: {total_rows} total rows")
        print(f"✓ Saved to: {output_path}")
        print(f"{'='*60}")
