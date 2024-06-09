from bs4 import BeautifulSoup
import re

# can add in the future 'search.listing.featured'
def get_offer_data(html):
    soup = BeautifulSoup(html, 'html.parser')
    offer_data_container = soup.find('div', attrs={'data-cy': 'search.listing.organic'})
    offers = []
    if offer_data_container is not None:
        for article in offer_data_container.find_all('article', {'data-cy': 'listing-item'}):
            offer_data = {}

            listing_item_header_div = article.find('div', attrs={'data-testid': 'listing-item-header'})
            if listing_item_header_div:
                price_span = listing_item_header_div.find('span')
                if price_span:
                    price_text = price_span.text.replace('\xa0', '').rstrip('zł').strip()
                    try:
                        offer_data['Price'] = int(price_text.replace(',', '.'))
                    except ValueError:
                        offer_data['Price'] = 'N/A'

                parent_div = listing_item_header_div.find_parent()
                if parent_div:
                    a_tag = parent_div.find('a', attrs={'data-testid': 'listing-item-link'})
                    if a_tag:
                        offer_data['Link'] = a_tag['href']
                        offer_data['Text'] = a_tag.text.strip()
                    else:
                        offer_data['Link'] = 'N/A'
                        offer_data['Text'] = 'N/A'
                    
                    location_p = parent_div.find('p', attrs={'data-testid': 'advert-card-address'})
                    if location_p:
                        location_parts = location_p.text.strip().split(',')
                        offer_data['Location'] = [part.strip() for part in location_parts]
                    else:
                        offer_data['Location'] = 'N/A'
                    
                    specs_list = parent_div.find('div', attrs={'data-testid': 'advert-card-specs-list'})
                    if specs_list:
                        specs = {}
                        for dt, dd in zip(specs_list.find_all('dt'), specs_list.find_all('dd')):
                            specs[dt.text.strip()] = re.sub(r'\xa0|zł/m²|m²', '', dd.text.strip()).strip()
                        offer_data['Specs'] = specs
            
            offers.append(offer_data)
    else:
        print('No offer data found in the provided HTML.')
    
    return offers

def get_offer_paged_data(html_list): 
    offers = []

    for html in html_list:
        offers.extend(get_offer_data(html))
    
    return offers