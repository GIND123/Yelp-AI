import React, { useState } from 'react';
import { createContext } from 'react';
import { Alert } from 'react-native';
export const AppContext = createContext();

function AppContextProvider({ children }) {
  const [isLoading, setIsLoading] = useState(false);
  const [business, setBusiness] = useState([]);

  function throwErrorAlert(main, info) {
    Alert.alert(main, info);
    return;
  }
  return (
    <AppContext value={{ isLoading, setIsLoading, business, setBusiness, throwErrorAlert }}>
      {children}
    </AppContext>
  );
}

export default AppContextProvider;
