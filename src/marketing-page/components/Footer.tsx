import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { Link as RouterLink } from 'react-router-dom';

const TB_RESOURCES = [
  { label: 'DOH National TB Program', href: 'https://ntp.doh.gov.ph/' },
  {
    label: 'DOH: Tuberculosis',
    href: 'https://doh.gov.ph/diseases/tuberculosis-disease/',
  },
  {
    label: 'WHO: Tuberculosis',
    href: 'https://www.who.int/news-room/fact-sheets/detail/tuberculosis',
  },
] as const;

export default function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        borderTop: '1px solid',
        borderColor: 'divider',
        mt: 4,
      }}
    >
      <Container
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', sm: 'row' },
          justifyContent: 'space-between',
          alignItems: { xs: 'center', sm: 'flex-start' },
          gap: 2,
          py: 3,
        }}
      >
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          © {new Date().getFullYear()} Chexit
        </Typography>
        <Stack spacing={0.75} alignItems={{ xs: 'center', sm: 'flex-end' }}>
          <Link
            component={RouterLink}
            to="/about"
            color="text.secondary"
            variant="body2"
            underline="hover"
          >
            About us
          </Link>
          {TB_RESOURCES.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
              color="text.secondary"
              variant="caption"
              underline="hover"
              sx={{ textAlign: { xs: 'center', sm: 'right' } }}
            >
              {item.label}
            </Link>
          ))}
        </Stack>
      </Container>
    </Box>
  );
}
