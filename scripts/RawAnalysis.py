import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


class RawAnalysis:
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads a dataset from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the loaded data.
        """
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
        """
        Computes and visualizes basic statistics about headline lengths.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing a 'headline' column.

        Returns
        -------
        None
        """
        try:
            df['headline_length'] = df['headline'].apply(len)
            print("📊 Headline Length Statistics:")
            print(df['headline_length'].describe())

            plt.figure(figsize=(10, 5))
            sns.histplot(df['headline_length'], bins=30, kde=True)
            plt.title("Headline Length Distribution")
            plt.xlabel("Length (characters)")
            plt.ylabel("Frequency")
            plt.show()

        except KeyError:
            print("❌ 'headline' column not found in the DataFrame.")
        except Exception as e:
            print(f"❌ An error occurred during headline analysis: {e}")
    def character_stats(self, headlines):
            """
            Compute and print basic statistics for headline character lengths.

            Args:
                headlines (pd.Series): A pandas Series containing headline strings.

            Raises:
                TypeError: If input is not a pandas Series.
                ValueError: If Series is empty.
            """
            if not isinstance(headlines, pd.Series):
                raise TypeError("Input must be a pandas Series.")

            if headlines.empty:
                raise ValueError("Headline Series is empty.")

            # Convert to string and handle missing values
            headlines = headlines.astype(str).fillna("")
            char_lengths = headlines.str.len()

            try:
                print("Character Length Statistics:")
                print(f"Mean: {char_lengths.mean():.2f}")
                print(f"Median: {char_lengths.median()}")
                print(f"Min: {char_lengths.min()}")
                print(f"Max: {char_lengths.max()}")
                print(f"Standard Deviation: {char_lengths.std():.2f}")
            except Exception as e:
                print(f"An error occurred while calculating stats: {e}")
    def articles_per_publisher(self, df: pd.DataFrame) -> None:
        """
        Counts and visualizes the number of articles per publisher.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing a 'publisher' column.

        Returns
        -------
        None
        """
        try:
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

        except KeyError:
            print("❌ 'publisher' column not found in the DataFrame.")
        except Exception as e:
            print(f"❌ An error occurred during publisher analysis: {e}")

    def publication_trends(self, df: pd.DataFrame) -> None:
        """
        Analyzes and visualizes article publication trends over time and by weekday.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing a 'date' column.

        Returns
        -------
        None
        """
        try:
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

        except KeyError:
            print("❌ 'date' column not found in the DataFrame.")
        except Exception as e:
            print(f"❌ An error occurred during publication trend analysis: {e}")

    def time_series_analysis(self, df, datetime_col='date'):
            """
            Analyze publication frequency over time and by hour of day.

            Args:
                df (pd.DataFrame): DataFrame with a datetime column.
                datetime_col (str): Column name containing datetime information.

            Returns:
                dict: Contains daily and hourly frequency series.
            """
            try:
                if datetime_col not in df.columns:
                    raise ValueError(f"'{datetime_col}' column not found in DataFrame.")

                # Convert to datetime and drop invalid
                df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
                df = df.dropna(subset=[datetime_col])

                # Daily frequency
                daily_counts = df.groupby(df[datetime_col].dt.date).size()

                # Hourly frequency
                # df['hour'] = df[datetime_col].dt.hour
                df.loc[:, 'hour'] = df[datetime_col].dt.hour
                hourly_counts = df.groupby('hour').size()

                # Plotting
                plt.figure(figsize=(12, 4))
                daily_counts.plot(title="Articles Published Per Day")
                plt.xlabel("Date")
                plt.ylabel("Article Count")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(8, 4))
                hourly_counts.plot(kind='bar', title="Articles by Hour of Day", xlabel='Hour', ylabel='Count')
                plt.grid(True)
                plt.tight_layout()
                plt.show()

                return {
                    "daily_counts": daily_counts,
                    "hourly_counts": hourly_counts
                }

            except Exception as e:
                print(f"Error in time_series_analysis: {e}")
                return {}
            
    def detect_publication_spikes(self, df, datetime_col='date', threshold=2.0):
        """
        Detect days with spikes in article publication volume based on Z-score.

        Args:
            df (pd.DataFrame): DataFrame with a datetime column.
            datetime_col (str): Name of the column containing datetime data.
            threshold (float): Z-score threshold for spike detection.

        Returns:
            pd.DataFrame: DataFrame with spike dates and article counts.
        """
        try:
            df = df.copy()
            if datetime_col not in df.columns:
                raise ValueError(f"'{datetime_col}' column not found in DataFrame.")

            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
            df = df.dropna(subset=[datetime_col])

            daily_counts = df.groupby(df[datetime_col].dt.date).size()
            mean = daily_counts.mean()
            std = daily_counts.std()
            z_scores = (daily_counts - mean) / std

            spikes = daily_counts[z_scores > threshold]

            # Plot
            plt.figure(figsize=(12, 4))
            daily_counts.plot(label="Daily Count")
            spikes.plot(style='ro', label="Spike", markersize=6)
            plt.axhline(mean + threshold * std, color='red', linestyle='--', label='Spike Threshold')
            plt.title("Spike Detection in Article Publications")
            plt.xlabel("Date")
            plt.ylabel("Article Count")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            return spikes.reset_index(name='article_count')

        except Exception as e:
            print(f"Error in detect_publication_spikes: {e}")
            return pd.DataFrame()

    def articles_per_year(self, df, datetime_col='date'):
        """
        Calculate and visualize the number of articles published per year.

        Args:
            df (pd.DataFrame): DataFrame containing a datetime column.
            datetime_col (str): Name of the datetime column.

        Returns:
            pd.Series: Yearly count of published articles.
        """
        try:
            df = df.copy()
            if datetime_col not in df.columns:
                raise ValueError(f"'{datetime_col}' column not found in DataFrame.")

            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
            df = df.dropna(subset=[datetime_col])

            df['year'] = df[datetime_col].dt.year
            yearly_counts = df['year'].value_counts().sort_index()

            # Plot
            yearly_counts.plot(kind='bar', figsize=(10, 5), color='skyblue')
            plt.title("Number of Articles Published Per Year")
            plt.xlabel("Year")
            plt.ylabel("Number of Articles")
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()

            return yearly_counts

        except Exception as e:
            print(f"Error in articles_per_year: {e}")
            return pd.Series()
    def publishing_time_distribution(self, df, datetime_col='date'):
            """
            Analyze and visualize the distribution of publishing times by hour.

            Args:
                df (pd.DataFrame): DataFrame containing a datetime column.
                datetime_col (str): Name of the datetime column to analyze.

            Returns:
                pd.Series: Hourly distribution of article counts.
            """
            try:
                df = df.copy()

                if datetime_col not in df.columns:
                    raise ValueError(f"'{datetime_col}' column not found.")

                df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
                df = df.dropna(subset=[datetime_col])

                df['hour'] = df[datetime_col].dt.hour
                hour_dist = df['hour'].value_counts().sort_index()

                # Plotting
                plt.figure(figsize=(10, 5))
                sns.barplot(x=hour_dist.index, y=hour_dist.values, palette="viridis")
                plt.title("Distribution of Article Publishing Times by Hour")
                plt.xlabel("Hour of Day (0–23)")
                plt.ylabel("Number of Articles")
                plt.xticks(range(0, 24))
                plt.grid(axis='y')
                plt.tight_layout()
                plt.show()

                return hour_dist

            except Exception as e:
                print(f"Error in publishing_time_distribution: {e}")
                return pd.Series()
            
    def publisher_contribution_analysis(self, df, publisher_col='publisher', headline_col='headline', top_n=10):
        """
        Analyze which publishers contribute most and explore differences in the type of news reported.

        Parameters:
        - df: pandas DataFrame containing the data.
        - publisher_col: Name of the column containing publisher names.
        - headline_col: Name of the column containing news headlines.
        - top_n: Number of top publishers to analyze.

        Returns:
        - DataFrame with article counts by publisher.
        - Bar plot of top contributing publishers.
        - Top keywords per top publisher using bag-of-words.
        """
        try:
            # Drop rows with missing values
            df = df.dropna(subset=[publisher_col, headline_col])
            
            # Count articles per publisher
            publisher_counts = df[publisher_col].value_counts().head(top_n)
            print(f"Top {top_n} publishers:\n", publisher_counts)

            # Plotting
            plt.figure(figsize=(12, 6))
            sns.barplot(x=publisher_counts.index, y=publisher_counts.values, palette="Set2")
            plt.title(f"Top {top_n} Publishers by Article Count")
            plt.xlabel("Publisher")
            plt.ylabel("Number of Articles")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # Keyword analysis using CountVectorizer for top publishers
            vectorizer = CountVectorizer(stop_words='english', max_features=100)

            for pub in publisher_counts.index:
                print(f"\n🔹 Top Keywords for Publisher: {pub}")
                pub_headlines = df[df[publisher_col] == pub][headline_col].astype(str)
                X = vectorizer.fit_transform(pub_headlines)
                keywords = vectorizer.get_feature_names_out()
                word_freq = X.sum(axis=0).A1
                keyword_freq = dict(zip(keywords, word_freq))
                sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                for word, freq in sorted_keywords:
                    print(f"{word}: {freq}")

            return publisher_counts.to_frame(name='article_count')

        except Exception as e:
            print("Error in publisher_contribution_analysis:", str(e))

    def publisher_domain_analysis(self, df, publisher_col='publisher'):
        """
        Analyze publisher domains if publisher names are email addresses.

        Parameters:
        - df: pandas DataFrame containing the data.
        - publisher_col: Name of the column containing email-like publisher values.

        Returns:
        - DataFrame with article counts by email domain.
        - Bar plot of top contributing domains.
        """
        try:
            # Drop missing values in publisher column
            df = df.dropna(subset=[publisher_col])

            # Extract domain from email (if email format)
            df['domain'] = df[publisher_col].str.extract(r'@([\w\.-]+)', expand=False)

            # Drop NaN domains (non-email entries)
            domain_counts = df['domain'].value_counts().dropna().head(10)

            # Display top domains
            print("Top publisher domains:\n", domain_counts)

            # Plotting
            plt.figure(figsize=(10, 5))
            sns.barplot(x=domain_counts.index, y=domain_counts.values, palette='Blues_d')
            plt.title("Top Publisher Domains (by Email Address)")
            plt.xlabel("Domain")
            plt.ylabel("Number of Articles")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            return domain_counts.to_frame(name='article_count')

        except Exception as e:
            print("Error in publisher_domain_analysis:", str(e))

    