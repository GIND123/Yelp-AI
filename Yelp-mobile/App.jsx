import React, { useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import RootStack from './navigation';
import AppContextProvider from './context/AppContext';

export default function App() {
  return (
    <AppContextProvider>
      <NavigationContainer>
        <RootStack />
      </NavigationContainer>
    </AppContextProvider>
  );
}
