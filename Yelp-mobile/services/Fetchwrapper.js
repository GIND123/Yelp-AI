export default async function FetchWrapper(URL, object) {
  try {
    let response = await fetch(URL, { ...object });
    let responsebody = await response.json();
    return responsebody;
  } catch (error) {
    if (error) return { success: false, message: error.message, statusCode: 500 };
  }
}
