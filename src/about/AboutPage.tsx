import * as React from 'react';
import Avatar from '@mui/material/Avatar';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Container from '@mui/material/Container';
import CssBaseline from '@mui/material/CssBaseline';
import Divider from '@mui/material/Divider';
import Stack from '@mui/material/Stack';
import { useColorScheme, useTheme } from '@mui/material/styles';
import Typography from '@mui/material/Typography';
import AppTheme from '../../shared-theme/AppTheme';
import AppAppBar from '../marketing-page/components/AppAppBar';
import Footer from '../marketing-page/components/Footer';

type Person = {
  name: string;
  role: string;
  bio: string;
  initials: string;
  photoSrc: string;
};

const PROJECT = {
  title: 'Chexit: Chest X-ray Identification for Tuberculosis',
  capstoneLabel: 'Research Capstone Project',
  institution:
    'University of the Philippines Diliman · Electrical and Electronics Engineering Institute',
  description:
    'Chexit was developed as a clinical decision-support system for Tuberculosis (TB) screening ' +
    'using chest X-rays (CXRs), aligning with the Department of Health’s (DOH) TB Elimination Plan ' +
    'under the strategic goal of incorporating new technologies for mass screening. The system ' +
    'outputs a TB probability score, a risk label, and a heatmap that explains which regions helped ' +
    'the AI calculate the assessment.',
  tags: [
    'Deep learning',
    'Chest X-ray',
    'Tuberculosis',
    'Explainable AI',
    'Medical AI',
    'Imaging',
  ],
};

const SDG_GOALS = [
  {
    number: 3,
    title: 'Good Health and Well-being',
    description: 'Ensure healthy lives and promote well-being for all ages',
    iconUrl: 'https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-03.jpg',
  },
  {
    number: 9,
    title: 'Industry, Innovation and Infrastructure',
    description:
      'Build resilient infrastructure, promote sustainable industrialization, and foster innovation',
    iconUrl: 'https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-09.jpg',
  },
] as const;

function useResolvedColorScheme(): 'light' | 'dark' {
  const { mode, systemMode } = useColorScheme();
  if (mode === 'system') {
    return systemMode === 'dark' ? 'dark' : 'light';
  }
  if (mode === 'dark') {
    return 'dark';
  }
  return 'light';
}

const AUTHORS: Person[] = [
  {
    name: 'Mark Joseph Garcia Ilagan',
    role: 'Co-author · BS Electronics Engineering',
    bio:
      'Mark Ilagan has academic and project experience in artificial intelligence, machine learning, and ' +
      'electronics engineering. He has developed a strong foundation in technical problem-solving, data ' +
      'analysis, and intelligent systems through academic work, internships, and applied projects. Beyond ' +
      'his university studies, he has gained international academic exposure in South Korea, Japan, ' +
      'Singapore, and KAIST, as well as industry experience in analytics and AI/ML development through ' +
      'Shopee Philippines.',
    initials: 'MI',
    photoSrc: '/assets/team/ilagan.jpg',
  },
  {
    name: 'Ma. Regina Rosel Manlutac Galfo',
    role: 'Co-author · BS Computer Engineering',
    bio:
      'Regina Galfo is an aspiring software engineer with deep interest in full-stack development, smart ' +
      'systems, and human-centered artificial intelligence. She has developed a strong aptitude for project ' +
      'leadership and a robust technical foundation through her engagements as event lead across recognized ' +
      'engineering organizations and her success as a champion of the Meralco IDOL Hackathon 2025 along with ' +
      'Mark. She has gained significant professional exposure as a student assistant for the Office of the ' +
      'Vice Chancellor for Academic Affairs, a cross-functional lead (branding & automation) for an Australian ' +
      'company, and most recently, as a software consultant for a Canadian-based IT company.',
    initials: 'RG',
    photoSrc: '/assets/team/galfo.jpg',
  },
];

const ADVISER: Person = {
  name: 'Dr. Jhoanna Rhodette I. Pedrasa',
  role:
    'EEEI · PhD Electrical Engineering, University of New South Wales, 2011',
  bio:
    'Dr. Jhoanna Pedrasa is a professor in ' +
    'the Electrical and Electronics Engineering Institute and has served as the College Secretary of the ' +
    'College of Engineering. Her research interests include computer networks, wireless sensor networks, and ' +
    'the design of experiments. Her work focuses on data analytics and modeling, artificial intelligence, and ' +
    'smart cities. She advises graduate and undergraduate researchers who are interested in tackling practical ' +
    'challenges in digital safety, sustainable infrastructure, inclusive technology, and other topics related ' +
    'to smart systems.',
  initials: 'JP',
  photoSrc: '/assets/team/pedrasa.jpg',
};

const EXAMINER: Person = {
  name: 'Dr. Jordan Rel C. Orillaza',
  role:
    'EEEI · PhD Electrical and Electronic Engineering, University of Canterbury, 2012',
  bio:
    'Dr. Jordan Orillaza’s ' +
    'research interests include harmonic stability and other topics related to power quality. Recently, he ' +
    'inherited the Electric Market Research Laboratory (from Dr. Nerves) and has shifted his research to ' +
    'electricity markets propped with his experiences from the Philippine Electricity Market Corporation ' +
    'Technical Committee. He currently advises graduate and undergraduate researchers who are keen to explore ' +
    'the participation of distributed resources in the electricity market and on processes that involve both ' +
    'the market operator and the system operator.',
  initials: 'JO',
  photoSrc: '/assets/team/orillaza.jpg',
};

type Institution = {
  name: string;
  initials: string;
  logoSrc: string;
  /** Alternate logo for dark mode (e.g. white mark on transparent). */
  logoSrcDark?: string;
  /** Optional local asset if the primary URL cannot be embedded (e.g. Facebook pages). */
  fallbackSrc?: string;
  /** White or light logos on a dark tile (e.g. SSL inverted banner). */
  logoOnDark?: boolean;
  /** Black or dark logos on a light tile (e.g. James Dyson Foundation). */
  logoOnLight?: boolean;
};

const ACKNOWLEDGEMENTS = {
  intro:
    'With the help of the following institutions, mentors, and collaborators who provided datasets, ' +
    'computing resources, clinical guidance, and feedback throughout the project — thank you.',
  institutions: [
    {
      name: 'University of the Philippines Diliman University Health Service',
      initials: 'UHS',
      logoSrc: '/assets/acknowledgements/UHS.png',
    },
    {
      name: 'University of the Philippines Philippine General Hospital',
      initials: 'PGH',
      logoSrc: '/assets/acknowledgements/pgh.png',
    },
    {
      name: 'The James Dyson Foundation',
      initials: 'JDF',
      logoSrc: '/assets/acknowledgements/dyson.png',
      logoSrcDark: '/assets/acknowledgements/dysonwhite.png',
    },
    {
      name: 'UP College of Engineering · Associate Dean for Student Affairs (COE ADSA)',
      initials: 'ADSA',
      logoSrc: '/assets/acknowledgements/ADSA.jpg',
    },
    {
      name: 'University of the Philippines Electrical and Electronics Engineering Institute',
      initials: 'EEEI',
      logoSrc: '/assets/acknowledgements/eeei.png',
    },
    {
      name: 'University of the Philippines Diliman Smart Systems Laboratory',
      initials: 'SSL',
      logoSrc: '/assets/acknowledgements/ssl.png',
      logoOnDark: true,
    },
    {
      name: 'University of the Philippines Diliman Data Protection Office',
      initials: 'DPO',
      logoSrc: '/assets/acknowledgements/dpo.png',
    },
  ] satisfies Institution[],
};

function SdgGoalItem({
  number,
  title,
  description,
  iconUrl,
}: {
  number: number;
  title: string;
  description: string;
  iconUrl: string;
}) {
  return (
    <Stack direction="row" spacing={1.5} alignItems="flex-start">
      <Box
        component="img"
        src={iconUrl}
        alt={`UN Sustainable Development Goal ${number}: ${title}`}
        loading="lazy"
        sx={{
          width: 48,
          height: 48,
          borderRadius: 0.75,
          flexShrink: 0,
          objectFit: 'cover',
        }}
      />
      <Box sx={{ minWidth: 0 }}>
        <Typography variant="caption" sx={{ fontWeight: 700, lineHeight: 1.35, display: 'block' }}>
          ({number}) {title}
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', mt: 0.25, lineHeight: 1.45 }}>
          {description}
        </Typography>
      </Box>
    </Stack>
  );
}

function InstitutionLogo({
  name,
  initials,
  logoSrc,
  logoSrcDark,
  fallbackSrc,
  logoOnDark = false,
  logoOnLight = false,
}: Institution) {
  const theme = useTheme();
  const resolvedScheme = useResolvedColorScheme();
  const isDarkMode = resolvedScheme === 'dark';
  const displaySrc = isDarkMode && logoSrcDark ? logoSrcDark : logoSrc;
  const useDarkTile = logoOnDark;
  const useLightTile = logoOnLight;

  const [imgSrc, setImgSrc] = React.useState(displaySrc);
  const [showFallback, setShowFallback] = React.useState(false);

  React.useEffect(() => {
    setImgSrc(displaySrc);
    setShowFallback(false);
  }, [displaySrc]);

  const handleError = () => {
    if (fallbackSrc && imgSrc !== fallbackSrc) {
      setImgSrc(fallbackSrc);
      return;
    }
    setShowFallback(true);
  };

  return (
    <Box
      aria-label={name}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        gap: 1,
        minHeight: 108,
        px: 1.5,
        py: 1.5,
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
        backgroundColor: useDarkTile
          ? '#0f172a'
          : useLightTile
            ? theme.palette.grey[50]
            : 'rgba(15,23,42,0.02)',
        color: 'text.secondary',
        transition: 'border-color 120ms ease, background-color 120ms ease',
        '&:hover': { borderColor: 'text.primary' },
        ...theme.applyStyles('dark', {
          backgroundColor: useDarkTile
            ? '#020617'
            : useLightTile
              ? '#f1f5f9'
              : 'rgba(255,255,255,0.02)',
        }),
      }}
    >
      {showFallback ? (
        <Typography variant="caption" sx={{ fontWeight: 700, letterSpacing: 1.5, fontSize: 12 }}>
          {initials}
        </Typography>
      ) : (
        <Box
          component="img"
          key={imgSrc}
          src={imgSrc}
          alt=""
          loading="lazy"
          referrerPolicy="no-referrer"
          onError={handleError}
          sx={{
            display: 'block',
            maxWidth: '100%',
            maxHeight: 56,
            width: 'auto',
            height: 'auto',
            objectFit: 'contain',
            flexShrink: 0,
          }}
        />
      )}
      <Typography variant="caption" sx={{ fontSize: 11, lineHeight: 1.25 }}>
        {name}
      </Typography>
    </Box>
  );
}

function PersonCard({
  person,
  accentColor = 'primary.main',
}: {
  person: Person;
  accentColor?: string;
}) {
  return (
    <Card
      variant="outlined"
      sx={{
        borderRadius: 3,
        height: '100%',
        transition: 'border-color 120ms ease, box-shadow 120ms ease',
        '&:hover': {
          borderColor: 'text.primary',
        },
      }}
    >
      <CardContent sx={{ p: { xs: 3, sm: 3.5 } }}>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 1.5 }}>
          <Avatar
            src={person.photoSrc}
            alt={person.name}
            imgProps={{ loading: 'lazy' }}
            sx={{
              bgcolor: accentColor,
              width: 56,
              height: 56,
              fontWeight: 700,
              fontSize: 18,
            }}
          >
            {person.initials}
          </Avatar>
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="h6" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
              {person.name}
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.25 }}>
              {person.role}
            </Typography>
          </Box>
        </Stack>
        <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1.65 }}>
          {person.bio}
        </Typography>
      </CardContent>
    </Card>
  );
}

export default function AboutPage(props: { disableCustomTheme?: boolean }) {
  return (
    <AppTheme {...props}>
      <CssBaseline enableColorScheme />
      <AppAppBar />
      <Box
        sx={(theme) => ({
          width: '100%',
          backgroundRepeat: 'no-repeat',
          backgroundImage:
            'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 90%), transparent)',
          ...theme.applyStyles('dark', {
            backgroundImage:
              'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 16%), transparent)',
          }),
        })}
      >
        <Container maxWidth="lg" sx={{ pt: { xs: 12, sm: 16 }, pb: { xs: 4, sm: 6 } }}>
          <Stack
            spacing={2}
            useFlexGap
            sx={{ alignItems: 'center', textAlign: 'center', maxWidth: 760, mx: 'auto' }}
          >
            <Typography
              variant="overline"
              sx={{ color: 'primary.main', letterSpacing: 2, fontWeight: 600 }}
            >
              ABOUT US
            </Typography>
            <Typography
              variant="h1"
              sx={(theme) => ({
                fontSize: 'clamp(2.4rem, 7vw, 3rem)',
                lineHeight: 1.1,
                color: 'primary.main',
                ...theme.applyStyles('dark', { color: 'primary.light' }),
              })}
            >
              Chexit
            </Typography>
            <Typography
              sx={{
                textAlign: 'center',
                color: 'text.secondary',
                width: { sm: '100%', md: '90%' },
              }}
            >
              A research capstone project building AI-assisted tuberculosis screening from chest
              X-rays. This page describes the work and the people behind it.
            </Typography>
          </Stack>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ pb: { xs: 6, md: 8 } }}>
        <Card variant="outlined" sx={{ borderRadius: 3, mb: { xs: 4, md: 6 } }}>
          <CardContent sx={{ p: { xs: 3, sm: 4 } }}>
            <Typography variant="overline" sx={{ color: 'text.secondary', letterSpacing: 1.5 }}>
              PROJECT
            </Typography>
            <Typography
              variant="h5"
              sx={{ fontWeight: 700, mt: 0.75, mb: 1.5, lineHeight: 1.25 }}
            >
              {PROJECT.title}
            </Typography>
            <Stack
              direction="row"
              spacing={1}
              useFlexGap
              flexWrap="wrap"
              sx={{ mb: 2.5, rowGap: 1 }}
            >
              <Chip label={PROJECT.capstoneLabel} size="small" color="primary" variant="outlined" />
              <Chip label={PROJECT.institution} size="small" />
            </Stack>
            <Typography variant="body1" sx={{ color: 'text.secondary', lineHeight: 1.75 }}>
              {PROJECT.description}
            </Typography>
            <Stack
              direction="row"
              spacing={0.75}
              sx={{ mt: 2.5, flexWrap: 'wrap', gap: 0.75 }}
            >
              {PROJECT.tags.map((tag) => (
                <Chip key={tag} label={tag} size="small" variant="outlined" />
              ))}
            </Stack>
          </CardContent>
        </Card>

        <Box sx={{ mb: { xs: 4, md: 6 }, px: { xs: 0.5, sm: 1 } }}>
          <Typography
            variant="overline"
            sx={{ color: 'text.secondary', letterSpacing: 1.5, fontSize: 11 }}
          >
            SUSTAINABLE DEVELOPMENT GOALS
          </Typography>
          <Box
            sx={{
              mt: 1.5,
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' },
              gap: { xs: 2, sm: 2.5 },
              maxWidth: 720,
            }}
          >
            {SDG_GOALS.map((goal) => (
              <SdgGoalItem key={goal.number} {...goal} />
            ))}
          </Box>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="overline" sx={{ color: 'text.secondary', letterSpacing: 1.5 }}>
            WHO WE ARE
          </Typography>
          <Typography variant="h4" sx={{ fontWeight: 700, mt: 0.5 }}>
            Meet the team behind Chexit
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
            This research capstone was researched and built by two students under the guidance of
            one faculty adviser, with the defense reviewed by an examiner.
          </Typography>
        </Box>

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' },
            gap: { xs: 2, sm: 2.5 },
            mt: 3,
          }}
        >
          {AUTHORS.map((author) => (
            <PersonCard key={author.name} person={author} />
          ))}
        </Box>

        <Divider sx={{ my: { xs: 4, sm: 5 } }}>
          <Chip label="Adviser" size="small" sx={{ px: 1 }} />
        </Divider>

        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Box sx={{ width: '100%', maxWidth: 540 }}>
            <PersonCard person={ADVISER} accentColor="secondary.main" />
          </Box>
        </Box>

        <Divider sx={{ my: { xs: 4, sm: 5 } }}>
          <Chip label="Examiner" size="small" sx={{ px: 1 }} />
        </Divider>

        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Box sx={{ width: '100%', maxWidth: 540 }}>
            <PersonCard person={EXAMINER} accentColor="info.main" />
          </Box>
        </Box>

        <Box sx={{ mt: { xs: 6, sm: 8 } }}>
          <Typography variant="overline" sx={{ color: 'text.secondary', letterSpacing: 1.5 }}>
            ACKNOWLEDGEMENTS
          </Typography>
          <Typography variant="h4" sx={{ fontWeight: 700, mt: 0.5 }}>
            With the help of
          </Typography>
          <Typography
            variant="body1"
            sx={{ color: 'text.secondary', mt: 1.25, maxWidth: 760, lineHeight: 1.7 }}
          >
            {ACKNOWLEDGEMENTS.intro}
          </Typography>

          <Box
            sx={{
              mt: { xs: 3, sm: 4 },
              display: 'grid',
              gridTemplateColumns: {
                xs: 'repeat(2, 1fr)',
                sm: 'repeat(3, 1fr)',
                md: 'repeat(4, 1fr)',
              },
              gap: { xs: 1.5, sm: 2 },
            }}
          >
            {ACKNOWLEDGEMENTS.institutions.map((institution) => (
              <InstitutionLogo key={institution.name} {...institution} />
            ))}
          </Box>
        </Box>
      </Container>

      <Footer />
    </AppTheme>
  );
}
