import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ReactQueryDevtools } from 'react-query/devtools';
import { ErrorBoundary } from 'react-error-boundary';

// Components
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import ThreatOverview from './pages/ThreatOverview';
import NetworkTopology from './pages/NetworkTopology';
import ThreatIntelligence from './pages/ThreatIntelligence';
import BehavioralAnalysis from './pages/BehavioralAnalysis';
import AttackCorrelation from './pages/AttackCorrelation';
import GeospatialAnalysis from './pages/GeospatialAnalysis';
import AdvancedAnalytics from './pages/AdvancedAnalytics';
import IncidentManagement from './pages/IncidentManagement';
import ReportsExport from './pages/ReportsExport';
import Settings from './pages/Settings';
import ErrorFallback from './components/Common/ErrorFallback';
import LoadingSpinner from './components/Common/LoadingSpinner';

// Hooks and Utils
import { useWebSocket } from './hooks/useWebSocket';
import { useTheme } from './hooks/useTheme';
import { useAuth } from './hooks/useAuth';
import { NotificationProvider } from './contexts/NotificationContext';
import { AlertProvider } from './contexts/AlertContext';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { theme, toggleTheme } = useTheme();
  const { isAuthenticated, isLoading } = useAuth();
  const { isConnected, lastMessage } = useWebSocket('ws://localhost:8001/ws');

  // Create MUI theme
  const muiTheme = createTheme({
    palette: {
      mode: theme,
      primary: {
        main: '#1e3c72',
        light: '#4a6fa5',
        dark: '#0d1b36',
      },
      secondary: {
        main: '#2a5298',
        light: '#5a7bc8',
        dark: '#1a3568',
      },
      error: {
        main: '#dc3545',
      },
      warning: {
        main: '#ffc107',
      },
      success: {
        main: '#28a745',
      },
      background: {
        default: theme === 'dark' ? '#121212' : '#f5f5f5',
        paper: theme === 'dark' ? '#1e1e1e' : '#ffffff',
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      h4: {
        fontWeight: 600,
      },
      h5: {
        fontWeight: 600,
      },
      h6: {
        fontWeight: 600,
      },
    },
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: theme === 'dark' 
              ? '0 4px 6px rgba(0, 0, 0, 0.3)' 
              : '0 4px 6px rgba(0, 0, 0, 0.1)',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            textTransform: 'none',
            fontWeight: 600,
          },
        },
      },
    },
  });

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={muiTheme}>
          <CssBaseline />
          <NotificationProvider>
            <AlertProvider>
              <Router>
                <Box sx={{ display: 'flex', minHeight: '100vh' }}>
                  <Sidebar 
                    open={sidebarOpen} 
                    onToggle={() => setSidebarOpen(!sidebarOpen)}
                  />
                  
                  <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    <Header 
                      onMenuClick={() => setSidebarOpen(!sidebarOpen)}
                      onThemeToggle={toggleTheme}
                      theme={theme}
                      isConnected={isConnected}
                    />
                    
                    <Box 
                      component="main" 
                      sx={{ 
                        flexGrow: 1, 
                        p: 3,
                        backgroundColor: 'background.default',
                        minHeight: 'calc(100vh - 64px)',
                      }}
                    >
                      <Routes>
                        <Route path="/" element={<ThreatOverview />} />
                        <Route path="/network-topology" element={<NetworkTopology />} />
                        <Route path="/threat-intelligence" element={<ThreatIntelligence />} />
                        <Route path="/behavioral-analysis" element={<BehavioralAnalysis />} />
                        <Route path="/attack-correlation" element={<AttackCorrelation />} />
                        <Route path="/geospatial-analysis" element={<GeospatialAnalysis />} />
                        <Route path="/advanced-analytics" element={<AdvancedAnalytics />} />
                        <Route path="/incident-management" element={<IncidentManagement />} />
                        <Route path="/reports-export" element={<ReportsExport />} />
                        <Route path="/settings" element={<Settings />} />
                      </Routes>
                    </Box>
                  </Box>
                </Box>
              </Router>
            </AlertProvider>
          </NotificationProvider>
        </ThemeProvider>
        <ReactQueryDevtools initialIsOpen={false} />
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;