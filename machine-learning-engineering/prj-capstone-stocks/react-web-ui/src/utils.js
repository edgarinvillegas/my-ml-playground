import {useFetch} from "react-async";

const baseApiEndpoint = 'https://6pjyfcf6t1.execute-api.us-west-2.amazonaws.com/v1';

export const useStockFetch = (api, method='GET') => {
    return useFetch(`${baseApiEndpoint}/${api}`, {
        headers: { accept: "application/json" },
        method
    })
};
/*
const APIResult = () => {
  const { data, error } = useStockFetch('get-data?ticker=GLD&forecastMonths=2&lookbackMonths=6&skipUpload=1', 'post')
  if (error) return <p>{error.message}</p>
  if (data) return <p>{JSON.stringify(data)}</p>
  return null
}*/

export const apiFetch$ = async (api, method='GET') => {
    const rawResponse = await fetch(`${baseApiEndpoint}/${api}`, {
        method,
        headers: { 'Accept': 'application/json' }
    });
    return await rawResponse.json();
}

