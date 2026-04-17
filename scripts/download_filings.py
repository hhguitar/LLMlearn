from tesla_qa.downloader import SecDownloader


if __name__ == '__main__':
    downloader = SecDownloader()
    downloader.run(start_year=2021, end_year=2025)
