import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Platform } from 'react-native';

import Home from '../screens/Home';
import AddPhoto from '../screens/AddPhoto';
import ChooseRestaurantScreen from '../screens/ChooseRestaurantScreen';
import ScheduleOrderScreen from '../screens/OrderDetailsScreen';
import RestaurantSummaryScreen from '../screens/RestaurantSummaryScreen';

const Stack = createNativeStackNavigator();

const defaultScreenOptions = {
  headerStyle: {
    backgroundColor: '#FFFFFF',
    elevation: 4, // Android shadow
    shadowOpacity: 0.1, // iOS shadow
  },
  headerTitleStyle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1F2937',
  },
  headerTintColor: '#4CAF50',
  headerShadowVisible: true,
  headerBackTitleVisible: false,
  animation: Platform.OS === 'android' ? 'slide_from_right' : 'default',
};

function RootStack() {
  return (
    <Stack.Navigator screenOptions={defaultScreenOptions}>
      <Stack.Screen
        name="Home"
        component={Home}
        options={{
          headerShown: false, // Home handles its own header
        }}
      />
      <Stack.Screen
        name="AddPhoto"
        component={AddPhoto}
        options={{
          title: 'Add Your Food',
          headerTitleStyle: {
            fontSize: 20,
            fontWeight: '700',
            color: '#1F2937',
          },
        }}
      />
      <Stack.Screen
        name="ChooseRestaurant"
        component={ChooseRestaurantScreen}
        options={{
          title: 'Choose Restaurant',
          headerTitleStyle: {
            fontSize: 20,
            fontWeight: '700',
            color: '#1F2937',
          },
        }}
      />
      <Stack.Screen
        name="ScheduleOrderScreen"
        component={ScheduleOrderScreen}
        options={{
          title: 'Order Details',
          headerTitleStyle: {
            fontSize: 20,
            fontWeight: '700',
            color: '#1F2937',
          },
        }}
      />
      <Stack.Screen
        name="RestaurantSummaryScreen"
        component={RestaurantSummaryScreen}
        options={{
          title: 'Restaurant Details',
          headerTitleStyle: {
            fontSize: 20,
            fontWeight: '700',
            color: '#1F2937',
          },
        }}
      />
    </Stack.Navigator>
  );
}

export default RootStack;