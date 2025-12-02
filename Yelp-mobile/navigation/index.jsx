import { createNativeStackNavigator } from '@react-navigation/native-stack';

import Home from '../screens/Home';
import AddPhoto from '../screens/AddPhoto';
import ChooseRestaurantScreen from '../screens/ChooseRestaurantScreen';
import ScheduleOrderScreen from '../screens/OrderDetailsScreen';
import RestaurantSummaryScreen from '../screens/RestaurantSummaryScreen';

const Stack = createNativeStackNavigator();

function RootStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Home" component={Home} options={{
        title:"Onlyfoods"
      }}/>
      <Stack.Screen name="AddPhoto" component={AddPhoto}/>
      <Stack.Screen name="ChooseRestaurant" component={ChooseRestaurantScreen}/>
      <Stack.Screen name="ScheduleOrderScreen" component={ScheduleOrderScreen}/>
      <Stack.Screen name="RestaurantSummaryScreen" component={RestaurantSummaryScreen}/>
    </Stack.Navigator>
  );
}

export default RootStack