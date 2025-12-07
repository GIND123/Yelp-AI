import FetchWrapper from './Fetchwrapper';
import { throwErrorAlert } from '../utils/alert';

export async function Test() {
  let res = await FetchWrapper(`${process.env.EXPO_PUBLIC_API_URL_PIPELINE_1}/health`);
  console.log(res);
}

export async function SearchImage(image, user_query, location, Latitude, Longitude, date, time) {
  try {
    const formData = new FormData();
    if (image) {
      formData.append('image', {
        uri: image,
        name: 'IMG_0001.jpeg',
        type: 'image/jpeg',
      });
    }
    formData.append('Location', location);
    formData.append('user_query', user_query);
    formData.append('Latitude', Latitude);
    formData.append('Longitude', Longitude);
    formData.append('Date', date);
    formData.append('Time', time);
    formData.append('save_to_file', 'false');

    let response;
    if (image) {
      response = await fetch(`${process.env.EXPO_PUBLIC_API_URL_PIPELINE_1}/search-image`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    } else {
      response = await fetch(`${process.env.EXPO_PUBLIC_API_URL_PIPELINE_1}/search-caption`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    }
    return await response.json();
  } catch (error) {
    throwErrorAlert('Error', error);
  }
}

export async function getRestaurantDetails(URL) {
  let response = await fetch(`${process.env.EXPO_PUBLIC_API_URL_PIPELINE_2}/analyze-business`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(URL),
  });

  return await response.json();
}
