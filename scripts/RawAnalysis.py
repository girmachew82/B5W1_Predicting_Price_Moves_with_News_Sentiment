import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RawAnalysis:

    def load_data(self, filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Successfully loaded dataset: {filepath}")
            return df
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
        except pd.errors.ParserError:
            print(f"❌ Error parsing the file: {filepath}")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")

    def headline_length_stats(self, df: pd.DataFrame) -> None:
        df['headline_length'] = df['headline'].apply(len)
        print("📊 Headline Length Statistics:")
        print(df['headline_length'].describe())

        plt.figure(figsize=(10, 5))
        sns.histplot(df['headline_length'], bins=30, kde=True)
        plt.title("Headline Length Distribution")
        plt.xlabel("Length (characters)")
        plt.ylabel("Frequency")
        plt.show()

    def articles_per_publisher(self, df: pd.DataFrame) -> None:
        publisher_counts = df['publisher'].value_counts()
        print("📰 Top Publishers by Number of Articles:")
        print(publisher_counts.head(10))

        plt.figure(figsize=(12, 6))
        sns.barplot(x=publisher_counts.head(10).index, y=publisher_counts.head(10).values)
        plt.title("Top 10 Publishers by Article Count")
        plt.ylabel("Articles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def publication_trends(self, df: pd.DataFrame) -> None:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['weekday'] = df['date'].dt.day_name()

        daily_counts = df['date'].dt.date.value_counts().sort_index()
        weekday_counts = df['weekday'].value_counts().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )

        # Plot daily trends
        plt.figure(figsize=(14, 6))
        daily_counts.plot()
        plt.title("Articles Published per Day")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot weekday trends
        plt.figure(figsize=(10, 5))
        sns.barplot(x=weekday_counts.index, y=weekday_counts.values)
        plt.title("Articles by Weekday")
        plt.xlabel("Weekday")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.show()
