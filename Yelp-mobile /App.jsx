import React, { useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import RootStack from './navigation';
import AppContextProvider from './context/AppContext';
import SplashScreen from './screens/SplashScreen';

export default function App() {
  // const [showSplash, setShowSplash] = useState(true);

  // if (showSplash) {
  //   return <SplashScreen onFinish={() => setShowSplash(false)} />;
  // }

  return (
    <AppContextProvider>
      <NavigationContainer>
        <RootStack />
      </NavigationContainer>
    </AppContextProvider>
  );
}
